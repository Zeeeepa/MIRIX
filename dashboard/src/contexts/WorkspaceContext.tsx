import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import apiClient from '@/api/client';
import { useAuth } from '@/contexts/AuthContext';

export interface User {
  id: string;
  name: string;
  organization_id: string;
  status: string;
  timezone: string;
  created_at: string;
}

interface WorkspaceContextType {
  users: User[];
  selectedUser: User | null;
  setSelectedUser: (user: User) => void;
  refreshUsers: () => Promise<void>;
  createUser: (name: string) => Promise<User>;
  deleteUser: (userId: string) => Promise<void>;
  isLoading: boolean;
}

const WorkspaceContext = createContext<WorkspaceContextType | null>(null);

export const useWorkspace = () => {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error('useWorkspace must be used within a WorkspaceProvider');
  }
  return context;
};

export const WorkspaceProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user: clientUser } = useAuth(); // This is the Client (Admin) - includes admin_user_id
  const [searchParams] = useSearchParams();
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const selectedUserRef = useRef<User | null>(null);
  const requestedUserParam = searchParams.get('user') || searchParams.get('user_id');

  useEffect(() => {
    selectedUserRef.current = selectedUser;
  }, [selectedUser]);

  const normalizeUserToken = (value: string) => value.replace(/_/g, '-');

  const findUserFromParam = (userList: User[], param: string | null) => {
    if (!param) return null;
    const normalizedParam = normalizeUserToken(param);
    return (
      userList.find((user) => user.id === param || user.name === param) ||
      userList.find((user) => normalizeUserToken(user.id) === normalizedParam) ||
      userList.find((user) => normalizeUserToken(user.name) === normalizedParam) ||
      null
    );
  };

  const fetchUsers = async () => {
    if (!clientUser) return;
    setIsLoading(true);
    try {
      // The clientUser now includes admin_user_id from the backend
      const defaultUserId = clientUser.admin_user_id;
      
      // List all users
      const response = await apiClient.get('/users');
      const userList = response.data;
      setUsers(userList);
      
      const matchedUser = findUserFromParam(userList, requestedUserParam);
      const currentSelected = selectedUserRef.current;

      if (matchedUser && currentSelected?.id !== matchedUser.id) {
        setSelectedUser(matchedUser);
      } else if (!currentSelected && userList.length > 0) {
        // Auto-select the default user if none is selected yet
        const defaultUser = userList.find((u: User) => u.id === defaultUserId);
        if (defaultUser) {
          setSelectedUser(defaultUser);
        } else {
          // Fallback to first user if default not found
          setSelectedUser(userList[0]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch users", error);
    } finally {
      setIsLoading(false);
    }
  };

  const createUser = async (name: string) => {
    if (!clientUser) throw new Error("Not authenticated");
    
    const response = await apiClient.post('/users/create_or_get', {
      name: name,
    });
    
    const newUser = response.data;
    await fetchUsers(); // Refresh list
    setSelectedUser(newUser); // Switch to new user
    return newUser;
  };

  const deleteUser = async (userId: string) => {
    if (!clientUser) throw new Error("Not authenticated");
    
    await apiClient.delete(`/users/${userId}`);
    
    // If deleted user was selected, switch to another user
    if (selectedUser?.id === userId) {
      const remainingUsers = users.filter(u => u.id !== userId);
      if (remainingUsers.length > 0) {
        // Try to select the default user, otherwise first available
        const defaultUser = remainingUsers.find(u => u.id === clientUser.admin_user_id);
        setSelectedUser(defaultUser || remainingUsers[0]);
      } else {
        setSelectedUser(null);
      }
    }
    
    await fetchUsers(); // Refresh list
  };

  useEffect(() => {
    if (clientUser) {
      fetchUsers();
    }
  }, [clientUser]);

  useEffect(() => {
    if (!requestedUserParam || users.length === 0) return;
    const matchedUser = findUserFromParam(users, requestedUserParam);
    if (matchedUser && selectedUserRef.current?.id !== matchedUser.id) {
      setSelectedUser(matchedUser);
    }
  }, [requestedUserParam, users]);

  return (
    <WorkspaceContext.Provider value={{ 
      users, 
      selectedUser, 
      setSelectedUser, 
      refreshUsers: fetchUsers,
      createUser,
      deleteUser,
      isLoading 
    }}>
      {children}
    </WorkspaceContext.Provider>
  );
};
