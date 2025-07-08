#!/usr/bin/env python3
"""
Debug script to see what SQL is being generated for the database schema
"""

import sys
import os

# Add the mirix module to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

# Import the base class and models
from mirix.orm.sqlalchemy_base import Base

# Import all models to make sure they're registered
from mirix.orm import *

def main():
    print("=== PGLite Schema Debug ===")
    print(f"Found {len(Base.metadata.tables)} tables")
    
    for i, (table_name, table) in enumerate(Base.metadata.tables.items()):
        print(f"\n--- Table {i+1}: {table_name} ---")
        try:
            create_table_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            print(f"SQL Length: {len(create_table_sql)} characters")
            print("SQL:")
            print(create_table_sql)
            print()
            
            # Test if this is a particularly complex table
            if len(create_table_sql) > 1000:
                print("⚠️  This table has very long SQL - might cause issues")
            
        except Exception as e:
            print(f"❌ Error generating SQL for table {table_name}: {e}")
    
    print("\n=== End Debug ===")

if __name__ == "__main__":
    main() 