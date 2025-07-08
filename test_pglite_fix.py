#!/usr/bin/env python3
"""
Test script to verify PGlite block creation and IN operator fixes
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set PGlite environment variables
os.environ['MIRIX_USE_PGLITE'] = 'true'
os.environ['MIRIX_PGLITE_BRIDGE_URL'] = 'http://127.0.0.1:8001'
os.environ['MIRIX_PG_URI'] = ''

# Add the mirix module to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_pglite_blocks():
    """Test PGlite block creation and querying with IN operator"""
    
    print("=== Testing PGlite Block Creation and IN Operator ===")
    
    try:
        # Import after setting environment variables
        from mirix.server.server import SessionLocal
        from mirix.orm.block import Block as BlockModel
        from mirix.schemas.block import Block
        from mirix.services.block_manager import BlockManager
        from mirix.services.user_manager import UserManager
        
        # Create a session
        with SessionLocal() as session:
            print("✅ PGlite session created successfully")
            
            # Create user and block managers
            user_manager = UserManager()
            block_manager = BlockManager()
            
            # Get default user
            default_user = user_manager.create_default_user()
            print(f"✅ Default user created: {default_user.id}")
            
            # Create test blocks
            test_blocks = []
            for i in range(3):
                block = Block(
                    label=f"test_block_{i}",
                    value=f"Test content {i}",
                    limit=2000,
                    is_template=False
                )
                created_block = block_manager.create_or_update_block(block, actor=default_user)
                test_blocks.append(created_block)
                print(f"✅ Created block {i}: {created_block.id}")
            
            # Test the IN operator by querying multiple block IDs
            block_ids = [block.id for block in test_blocks]
            print(f"Testing IN operator with block IDs: {block_ids}")
            
            # This should trigger the IN operator in the filter method
            found_blocks = session.query(BlockModel).filter(BlockModel.id.in_(block_ids)).all()
            
            print(f"✅ Found {len(found_blocks)} blocks using IN operator")
            
            for block in found_blocks:
                if hasattr(block, 'to_pydantic'):
                    pydantic_block = block.to_pydantic()
                    print(f"  - Block: {pydantic_block.id}, Label: {pydantic_block.label}")
                else:
                    print(f"  - Block: {block.id}, Label: {block.label}")
            
            # Clean up test blocks
            for block in test_blocks:
                try:
                    block_manager.delete_block(block.id, actor=default_user)
                    print(f"✅ Cleaned up block: {block.id}")
                except Exception as e:
                    print(f"⚠️  Failed to clean up block {block.id}: {e}")
            
            print("✅ PGlite IN operator test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ PGlite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_pglite_blocks() 