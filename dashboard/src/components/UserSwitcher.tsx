import React, { useState } from 'react';
import { ChevronsUpDown, Check, Plus, User as UserIcon } from 'lucide-react';
import { useWorkspace, User } from '@/contexts/WorkspaceContext';
import { useAuth } from '@/contexts/AuthContext';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

export const UserSwitcher: React.FC<{ className?: string }> = ({ className }) => {
  const { users, selectedUser, setSelectedUser, createUser, isLoading } = useWorkspace();
  const { user: clientUser } = useAuth();
  const [open, setOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newUserName, setNewUserName] = useState('');
  const [loading, setLoading] = useState(false);

  // Check if a user is the default user
  const isDefaultUser = (user: User) => {
    return clientUser?.default_user_id === user.id;
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      await createUser(newUserName);
      setNewUserName('');
      setDialogOpen(false);
      setOpen(false);
    } catch (error) {
      console.error("Failed to create user", error);
    } finally {
      setLoading(false);
    }
  };

  // Get display name with "(Default)" suffix if applicable
  const getDisplayName = (user: User) => {
    if (isDefaultUser(user)) {
      return `${user.name} (Default)`;
    }
    return user.name;
  };

  return (
    <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className={cn("w-full justify-between", className)}
            disabled={isLoading}
          >
            <div className="flex items-center gap-2 truncate">
              <div className="h-5 w-5 rounded-full bg-primary/10 flex items-center justify-center text-primary text-xs font-bold">
                {selectedUser?.name.charAt(0).toUpperCase() || 'U'}
              </div>
              <span className="truncate">
                {isLoading ? 'Loading...' : (selectedUser ? getDisplayName(selectedUser) : 'Select User')}
              </span>
            </div>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[220px] p-0" align="start">
          <div className="max-h-[300px] overflow-y-auto p-1">
            <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
              Users
            </div>
            {users.map((user) => (
              <div
                key={user.id}
                className={cn(
                  "relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                  selectedUser?.id === user.id && "bg-accent text-accent-foreground"
                )}
                onClick={() => {
                  setSelectedUser(user);
                  setOpen(false);
                }}
              >
                <UserIcon className="mr-2 h-4 w-4 shrink-0" />
                <span className="truncate flex-1">{user.name}</span>
                {isDefaultUser(user) && (
                  <span className="ml-1 text-xs text-muted-foreground">(Default)</span>
                )}
                {selectedUser?.id === user.id && (
                  <Check className="ml-2 h-4 w-4 shrink-0" />
                )}
              </div>
            ))}
            <div className="border-t my-1" />
            <DialogTrigger asChild>
              <div
                className="relative flex cursor-pointer select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-accent hover:text-accent-foreground"
              >
                <Plus className="mr-2 h-4 w-4" />
                Create New User
              </div>
            </DialogTrigger>
          </div>
        </PopoverContent>
      </Popover>

      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create User</DialogTitle>
          <DialogDescription>
            Create a new user to associate memories and interactions with.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleCreateUser}>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="name" className="text-right">
                Name
              </Label>
              <Input
                id="name"
                value={newUserName}
                onChange={(e) => setNewUserName(e.target.value)}
                className="col-span-3"
                placeholder="e.g. John Doe"
                required
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="submit" disabled={loading}>
              {loading ? 'Creating...' : 'Create User'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
