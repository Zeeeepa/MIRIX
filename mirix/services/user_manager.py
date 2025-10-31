from typing import List, Optional, Tuple

from mirix.orm.errors import NoResultFound
from mirix.orm.organization import Organization as OrganizationModel
from mirix.orm.user import User as UserModel
from mirix.schemas.user import User as PydanticUser
from mirix.schemas.user import UserUpdate
from mirix.services.organization_manager import OrganizationManager
from mirix.utils import enforce_types


class UserManager:
    """Manager class to handle business logic related to Users."""

    DEFAULT_USER_NAME = "default_user"
    DEFAULT_USER_ID = "user-00000000-0000-4000-8000-000000000000"
    DEFAULT_TIME_ZONE = "UTC (UTC+00:00)"

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from mirix.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_default_user(
        self, org_id: str = OrganizationManager.DEFAULT_ORG_ID
    ) -> PydanticUser:
        """Create the default user."""
        with self.session_maker() as session:
            # Make sure the org id exists
            try:
                OrganizationModel.read(db_session=session, identifier=org_id)
            except NoResultFound:
                raise ValueError(
                    f"No organization with {org_id} exists in the organization table."
                )

            # Try to retrieve the user
            try:
                user = UserModel.read(
                    db_session=session, identifier=self.DEFAULT_USER_ID
                )
            except NoResultFound:
                # If it doesn't exist, make it
                user = UserModel(
                    id=self.DEFAULT_USER_ID,
                    name=self.DEFAULT_USER_NAME,
                    status="active",
                    timezone=self.DEFAULT_TIME_ZONE,
                    organization_id=org_id,
                )
                user.create(session)

            return user.to_pydantic()

    @enforce_types
    def create_user(self, pydantic_user: PydanticUser) -> PydanticUser:
        """Create a new user if it doesn't already exist (with Redis caching)."""
        with self.session_maker() as session:
            new_user = UserModel(**pydantic_user.model_dump())
            new_user.create_with_redis(session, actor=None)  # ⭐ Auto-caches to Redis
            return new_user.to_pydantic()

    @enforce_types
    def update_user(self, user_update: UserUpdate) -> PydanticUser:
        """Update user details (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(
                db_session=session, identifier=user_update.id
            )

            # Update only the fields that are provided in UserUpdate
            update_data = user_update.model_dump(exclude_unset=True, exclude_none=True)
            for key, value in update_data.items():
                setattr(existing_user, key, value)

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # ⭐ Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_timezone(self, timezone_str: str, user_id: str) -> PydanticUser:
        """Update the timezone of a user (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the timezone
            existing_user.timezone = timezone_str

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # ⭐ Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def update_user_status(self, user_id: str, status: str) -> PydanticUser:
        """Update the status of a user (with Redis cache invalidation)."""
        with self.session_maker() as session:
            # Retrieve the existing user by ID
            existing_user = UserModel.read(db_session=session, identifier=user_id)

            # Update the status
            existing_user.status = status

            # Commit the updated user and update cache
            existing_user.update_with_redis(session, actor=None)  # ⭐ Updates Redis cache
            return existing_user.to_pydantic()

    @enforce_types
    def delete_user_by_id(self, user_id: str):
        """Delete a user and their associated records (removes from Redis cache)."""
        with self.session_maker() as session:
            # Delete from user table
            user = UserModel.read(db_session=session, identifier=user_id)
            
            # Remove from Redis cache before hard delete
            try:
                from mirix.database.redis_client import get_redis_client
                from mirix.log import get_logger
                
                logger = get_logger(__name__)
                redis_client = get_redis_client()
                if redis_client:
                    redis_key = f"{redis_client.USER_PREFIX}{user_id}"
                    redis_client.delete(redis_key)
                    logger.debug("Removed user %s from Redis cache", user_id)
            except Exception as e:
                from mirix.log import get_logger
                logger = get_logger(__name__)
                logger.warning("Failed to remove user %s from Redis cache: %s", user_id, e)
            
            user.hard_delete(session)

            session.commit()

    @enforce_types
    def get_user_by_id(self, user_id: str) -> PydanticUser:
        """Fetch a user by ID (with Redis Hash caching)."""
        # Try Redis cache first
        try:
            from mirix.database.redis_client import get_redis_client
            from mirix.log import get_logger
            
            logger = get_logger(__name__)
            redis_client = get_redis_client()
            
            if redis_client:
                redis_key = f"{redis_client.USER_PREFIX}{user_id}"
                cached_data = redis_client.get_hash(redis_key)
                if cached_data:
                    logger.debug("✅ Redis cache HIT for user %s", user_id)
                    return PydanticUser(**cached_data)
        except Exception as e:
            # Log but continue to PostgreSQL on Redis error
            from mirix.log import get_logger
            logger = get_logger(__name__)
            logger.warning("Redis cache read failed for user %s: %s", user_id, e)
        
        # Cache MISS or Redis unavailable - fetch from PostgreSQL
        with self.session_maker() as session:
            user = UserModel.read(db_session=session, identifier=user_id)
            pydantic_user = user.to_pydantic()
            
            # Populate Redis cache for next time
            try:
                if redis_client:
                    from mirix.settings import settings
                    data = pydantic_user.model_dump(mode='json')
                    # model_dump(mode='json') already converts datetime to ISO format strings
                    redis_client.set_hash(redis_key, data, ttl=settings.redis_ttl_users)
                    logger.debug("Populated Redis cache for user %s", user_id)
            except Exception as e:
                logger.warning("Failed to populate Redis cache for user %s: %s", user_id, e)
            
            return pydantic_user

    @enforce_types
    def get_default_user(self) -> PydanticUser:
        """Fetch the default user."""
        return self.get_user_by_id(self.DEFAULT_USER_ID)

    @enforce_types
    def get_user_or_default(self, user_id: Optional[str] = None):
        """Fetch the user or default user."""
        if not user_id:
            return self.get_default_user()

        try:
            return self.get_user_by_id(user_id=user_id)
        except NoResultFound:
            return self.get_default_user()

    @enforce_types
    def list_users(
        self, cursor: Optional[str] = None, limit: Optional[int] = 50
    ) -> Tuple[Optional[str], List[PydanticUser]]:
        """List users with pagination using cursor (id) and limit."""
        with self.session_maker() as session:
            results = UserModel.list(db_session=session, cursor=cursor, limit=limit)
            return [user.to_pydantic() for user in results]
