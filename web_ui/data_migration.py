#!/usr/bin/env python3
"""
Data Migration Utility for AI Research Lab Desktop App

Handles migration of existing data to the new data manager structure.
"""

import os
import sys
import json
import sqlite3
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataMigration:
    """Handles migration of existing data to new structure."""
    
    def __init__(self, old_data_dir: str, new_data_manager):
        """Initialize migration with old and new data locations."""
        self.old_data_dir = Path(old_data_dir)
        self.data_manager = new_data_manager
        self.migration_log = []
    
    def migrate_database(self) -> bool:
        """Migrate existing database to new location."""
        try:
            old_db_path = self.old_data_dir / "research_sessions.db"
            if not old_db_path.exists():
                logger.info("No existing database found to migrate")
                return True
            
            logger.info(f"Migrating database from {old_db_path} to {self.data_manager.db_path}")
            
            # Create backup of old database
            backup_path = self.old_data_dir / "research_sessions_backup.db"
            shutil.copy2(old_db_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
            
            # Copy database to new location
            shutil.copy2(old_db_path, self.data_manager.db_path)
            logger.info(f"Database migrated to {self.data_manager.db_path}")
            
            # Verify migration
            if self._verify_database_migration():
                logger.info("Database migration completed successfully")
                self.migration_log.append({
                    'type': 'database',
                    'status': 'success',
                    'old_path': str(old_db_path),
                    'new_path': str(self.data_manager.db_path),
                    'backup_path': str(backup_path)
                })
                return True
            else:
                logger.error("Database migration verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            self.migration_log.append({
                'type': 'database',
                'status': 'error',
                'error': str(e)
            })
            return False
    
    def migrate_config(self) -> bool:
        """Migrate existing config to new location."""
        try:
            old_config_path = self.old_data_dir / "config.json"
            if not old_config_path.exists():
                logger.info("No existing config found to migrate")
                return True
            
            logger.info(f"Migrating config from {old_config_path} to {self.data_manager.config_path}")
            
            # Read old config
            with open(old_config_path, 'r') as f:
                old_config = json.load(f)
            
            # Create backup
            backup_path = self.old_data_dir / "config_backup.json"
            shutil.copy2(old_config_path, backup_path)
            logger.info(f"Created config backup at {backup_path}")
            
            # Copy to new location
            shutil.copy2(old_config_path, self.data_manager.config_path)
            logger.info(f"Config migrated to {self.data_manager.config_path}")
            
            self.migration_log.append({
                'type': 'config',
                'status': 'success',
                'old_path': str(old_config_path),
                'new_path': str(self.data_manager.config_path),
                'backup_path': str(backup_path)
            })
            return True
            
        except Exception as e:
            logger.error(f"Config migration failed: {e}")
            self.migration_log.append({
                'type': 'config',
                'status': 'error',
                'error': str(e)
            })
            return False
    
    def migrate_vector_database(self) -> bool:
        """Migrate vector database if it exists."""
        try:
            old_vector_db_path = self.old_data_dir / "memory" / "vector_memory.db"
            if not old_vector_db_path.exists():
                logger.info("No existing vector database found to migrate")
                return True
            
            logger.info(f"Migrating vector database from {old_vector_db_path} to {self.data_manager.vector_db_path}")
            
            # Create backup
            backup_path = self.old_data_dir / "memory" / "vector_memory_backup.db"
            shutil.copy2(old_vector_db_path, backup_path)
            logger.info(f"Created vector database backup at {backup_path}")
            
            # Copy to new location
            shutil.copy2(old_vector_db_path, self.data_manager.vector_db_path)
            logger.info(f"Vector database migrated to {self.data_manager.vector_db_path}")
            
            self.migration_log.append({
                'type': 'vector_database',
                'status': 'success',
                'old_path': str(old_vector_db_path),
                'new_path': str(self.data_manager.vector_db_path),
                'backup_path': str(backup_path)
            })
            return True
            
        except Exception as e:
            logger.error(f"Vector database migration failed: {e}")
            self.migration_log.append({
                'type': 'vector_database',
                'status': 'error',
                'error': str(e)
            })
            return False
    
    def migrate_output_files(self) -> bool:
        """Migrate output files to new structure."""
        try:
            old_output_dir = self.old_data_dir / "output"
            if not old_output_dir.exists():
                logger.info("No existing output directory found to migrate")
                return True
            
            new_output_dir = self.data_manager.data_dir / "output"
            new_output_dir.mkdir(exist_ok=True)
            
            logger.info(f"Migrating output files from {old_output_dir} to {new_output_dir}")
            
            # Copy all files from old output directory
            for file_path in old_output_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(old_output_dir)
                    new_file_path = new_output_dir / relative_path
                    new_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, new_file_path)
            
            logger.info("Output files migrated successfully")
            
            self.migration_log.append({
                'type': 'output_files',
                'status': 'success',
                'old_path': str(old_output_dir),
                'new_path': str(new_output_dir)
            })
            return True
            
        except Exception as e:
            logger.error(f"Output files migration failed: {e}")
            self.migration_log.append({
                'type': 'output_files',
                'status': 'error',
                'error': str(e)
            })
            return False
    
    def _verify_database_migration(self) -> bool:
        """Verify that database migration was successful."""
        try:
            # Check if new database exists
            if not self.data_manager.db_path.exists():
                return False
            
            # Try to connect and query the database
            conn = sqlite3.connect(self.data_manager.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['sessions', 'metrics', 'agent_performance', 'chat_logs', 'agent_activity', 'meetings']
            
            for table in expected_tables:
                if table not in tables:
                    logger.error(f"Expected table '{table}' not found in migrated database")
                    return False
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Database verification failed: {e}")
            return False
    
    def run_full_migration(self) -> Dict[str, Any]:
        """Run complete migration of all data."""
        logger.info("Starting full data migration...")
        
        migration_results = {
            'success': True,
            'migrations': [],
            'errors': []
        }
        
        # Run all migrations
        migrations = [
            ('database', self.migrate_database),
            ('config', self.migrate_config),
            ('vector_database', self.migrate_vector_database),
            ('output_files', self.migrate_output_files)
        ]
        
        for migration_name, migration_func in migrations:
            try:
                logger.info(f"Running {migration_name} migration...")
                success = migration_func()
                if success:
                    migration_results['migrations'].append({
                        'type': migration_name,
                        'status': 'success'
                    })
                else:
                    migration_results['migrations'].append({
                        'type': migration_name,
                        'status': 'failed'
                    })
                    migration_results['success'] = False
                    migration_results['errors'].append(f"{migration_name} migration failed")
                    
            except Exception as e:
                logger.error(f"{migration_name} migration error: {e}")
                migration_results['migrations'].append({
                    'type': migration_name,
                    'status': 'error',
                    'error': str(e)
                })
                migration_results['success'] = False
                migration_results['errors'].append(f"{migration_name} migration error: {e}")
        
        # Save migration log
        log_path = self.data_manager.logs_dir / f"migration_log_{int(time.time())}.json"
        with open(log_path, 'w') as f:
            json.dump({
                'migration_results': migration_results,
                'migration_log': self.migration_log
            }, f, indent=2)
        
        logger.info(f"Migration completed. Results: {migration_results['success']}")
        return migration_results
    
    def cleanup_old_data(self, keep_backups: bool = True) -> bool:
        """Clean up old data after successful migration."""
        try:
            logger.info("Cleaning up old data...")
            
            # List of files/directories to potentially remove
            cleanup_items = [
                self.old_data_dir / "research_sessions.db",
                self.old_data_dir / "config.json",
                self.old_data_dir / "output"
            ]
            
            for item in cleanup_items:
                if item.exists():
                    if keep_backups:
                        # Move to backup instead of deleting
                        backup_dir = self.old_data_dir / "migration_backup"
                        backup_dir.mkdir(exist_ok=True)
                        backup_path = backup_dir / item.name
                        shutil.move(str(item), str(backup_path))
                        logger.info(f"Moved {item} to {backup_path}")
                    else:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                        logger.info(f"Removed {item}")
            
            logger.info("Old data cleanup completed")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

def run_migration_from_web_ui():
    """Run migration from web UI directory."""
    from data_manager import DataManager
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Get current directory (web_ui)
    current_dir = Path(__file__).parent
    
    # Create migration instance
    migration = DataMigration(str(current_dir), data_manager)
    
    # Run migration
    results = migration.run_full_migration()
    
    return results

if __name__ == '__main__':
    import time
    
    # Run migration
    results = run_migration_from_web_ui()
    
    print("Migration Results:")
    print(json.dumps(results, indent=2))
    
    if results['success']:
        print("\nMigration completed successfully!")
    else:
        print("\nMigration completed with errors:")
        for error in results['errors']:
            print(f"  - {error}") 