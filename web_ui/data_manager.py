#!/usr/bin/env python3
"""
Data Manager for AI Research Lab Desktop App

Handles all data persistence operations including:
- Real-time data persistence for messages, meetings, agent activity
- Session continuity across app restarts
- Database management with backup and recovery
- Data export/import functionality
- Data validation and integrity checks
- Data archiving and cleanup
- Configuration management
"""

import os
import sys
import json
import sqlite3
import shutil
import zipfile
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class DataManager:
    """Manages all data persistence operations for the desktop app."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the data manager with proper directory structure."""
        if base_dir is None:
            # Use user's home directory for data storage
            self.base_dir = Path.home() / ".ai_research_lab"
        else:
            self.base_dir = Path(base_dir)
        
        # Create directory structure
        self.data_dir = self.base_dir / "data"
        self.backup_dir = self.base_dir / "backups"
        self.archive_dir = self.base_dir / "archives"
        self.config_dir = self.base_dir / "config"
        self.logs_dir = self.base_dir / "logs"
        
        # Database paths
        self.db_path = self.data_dir / "research_sessions.db"
        self.vector_db_path = self.data_dir / "vector_memory.db"
        self.config_path = self.config_dir / "config.json"
        
        # Create directories
        self._create_directories()
        
        # Initialize database
        self._init_database()
        
        # Start backup thread
        self._start_backup_thread()
    
    def _create_directories(self):
        """Create the necessary directory structure."""
        directories = [
            self.data_dir,
            self.backup_dir,
            self.archive_dir,
            self.config_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _init_database(self):
        """Initialize the SQLite database with proper schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                research_question TEXT,
                config TEXT,
                results TEXT,
                logs TEXT,
                metadata TEXT
            )
        ''')
        
        # Add metadata column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE sessions ADD COLUMN metadata TEXT')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Create system metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                active_agents INTEGER,
                session_id TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Create agent performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                task_completed INTEGER,
                quality_score REAL,
                status TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Create chat logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                log_type TEXT CHECK(log_type IN ('thought', 'choice', 'communication', 'tool_call', 'system')),
                author TEXT,
                message TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Create agent activity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                activity_type TEXT CHECK(activity_type IN ('thinking', 'speaking', 'tool_use', 'meeting', 'idle')),
                message TEXT,
                status TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Create meetings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                meeting_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                participants TEXT,
                topic TEXT,
                agenda TEXT,
                transcript TEXT,
                outcomes TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Add missing columns if they don't exist
        try:
            cursor.execute('ALTER TABLE meetings ADD COLUMN agenda TEXT')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute('ALTER TABLE meetings ADD COLUMN transcript TEXT')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute('ALTER TABLE meetings ADD COLUMN outcomes TEXT')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        try:
            cursor.execute('ALTER TABLE meetings ADD COLUMN metadata TEXT')
        except sqlite3.OperationalError:
            # Column already exists
            pass
        
        # Create data integrity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_integrity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                record_count INTEGER,
                last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                checksum TEXT,
                status TEXT DEFAULT 'valid'
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_logs_session_id ON chat_logs(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_activity_session_id ON agent_activity(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_activity_timestamp ON agent_activity(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_meetings_session_id ON meetings(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_meetings_timestamp ON meetings(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at: {self.db_path}")
    
    def get_db_connection(self):
        """Get a database connection with proper configuration."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # PERSISTENT STORAGE METHODS
    
    def persist_chat_log(self, session_id: str, log_type: str, author: str, message: str, metadata: Optional[Dict] = None) -> bool:
        """Persist a chat log entry to the database."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute('''
                INSERT INTO chat_logs (session_id, log_type, author, message, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, log_type, author, message, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Persisted chat log: {session_id} - {author}: {message[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting chat log: {e}")
            return False
    
    def persist_agent_activity(self, session_id: str, agent_id: str, activity_type: str, message: str, status: str = 'active', metadata: Optional[Dict] = None) -> bool:
        """Persist agent activity to the database."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute('''
                INSERT INTO agent_activity (session_id, agent_id, activity_type, message, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, agent_id, activity_type, message, status, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Persisted agent activity: {session_id} - {agent_id}: {activity_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting agent activity: {e}")
            return False
    
    def persist_meeting(self, session_id: str, meeting_id: str, participants: List[str], topic: str, agenda: Optional[Dict] = None, transcript: Optional[List] = None, outcomes: Optional[Dict] = None, metadata: Optional[Dict] = None) -> bool:
        """Persist a meeting to the database."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            participants_json = json.dumps(participants)
            agenda_json = json.dumps(agenda or {})
            transcript_json = json.dumps(transcript or [])
            outcomes_json = json.dumps(outcomes or {})
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute('''
                INSERT INTO meetings (session_id, meeting_id, participants, topic, agenda, transcript, outcomes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, meeting_id, participants_json, topic, agenda_json, transcript_json, outcomes_json, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Persisted meeting: {session_id} - {meeting_id}: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting meeting: {e}")
            return False
    
    def persist_session(self, session_id: str, research_question: str, status: str = 'pending', config: Optional[Dict] = None, results: Optional[Dict] = None, metadata: Optional[Dict] = None) -> bool:
        """Persist or update a research session."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            config_json = json.dumps(config or {})
            results_json = json.dumps(results or {})
            metadata_json = json.dumps(metadata or {})
            
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (id, research_question, status, config, results, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (session_id, research_question, status, config_json, results_json, metadata_json))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Persisted session: {session_id} - {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error persisting session: {e}")
            return False
    
    def persist_metrics(self, session_id: Optional[str], cpu_usage: float, memory_usage: float, active_agents: int) -> bool:
        """Persist system metrics to the database."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO metrics (session_id, cpu_usage, memory_usage, active_agents)
                VALUES (?, ?, ?, ?)
            ''', (session_id, cpu_usage, memory_usage, active_agents))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
            return False
    
    # DATA RETRIEVAL METHODS
    
    def get_all_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all research sessions."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, created_at, updated_at, status, research_question, config, results, metadata
                FROM sessions
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            sessions = []
            for row in cursor.fetchall():
                session = dict(row)
                # Parse JSON fields
                if session.get('config'):
                    session['config'] = json.loads(session['config'])
                if session.get('results'):
                    session['results'] = json.loads(session['results'])
                if session.get('metadata'):
                    session['metadata'] = json.loads(session['metadata'])
                sessions.append(session)
            
            conn.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session with all its data."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get session
            cursor.execute('''
                SELECT * FROM sessions WHERE id = ?
            ''', (session_id,))
            
            session_row = cursor.fetchone()
            if not session_row:
                conn.close()
                return None
            
            session = dict(session_row)
            
            # Parse JSON fields
            if session.get('config'):
                session['config'] = json.loads(session['config'])
            if session.get('results'):
                session['results'] = json.loads(session['results'])
            if session.get('metadata'):
                session['metadata'] = json.loads(session['metadata'])
            
            # Get chat logs
            cursor.execute('''
                SELECT * FROM chat_logs 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            session['chat_logs'] = [dict(row) for row in cursor.fetchall()]
            
            # Get agent activity
            cursor.execute('''
                SELECT * FROM agent_activity 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            session['agent_activity'] = [dict(row) for row in cursor.fetchall()]
            
            # Get meetings
            cursor.execute('''
                SELECT * FROM meetings 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            session['meetings'] = [dict(row) for row in cursor.fetchall()]
            
            # Get metrics
            cursor.execute('''
                SELECT * FROM metrics 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            ''', (session_id,))
            session['metrics'] = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return session
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    def get_chat_logs(self, session_id: Optional[str] = None, log_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chat logs with optional filtering."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM chat_logs WHERE 1=1'
            params = []
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            if log_type:
                query += ' AND log_type = ?'
                params.append(log_type)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            logs = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return logs
            
        except Exception as e:
            logger.error(f"Error getting chat logs: {e}")
            return []
    
    def get_agent_activity(self, session_id: Optional[str] = None, agent_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get agent activity with optional filtering."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM agent_activity WHERE 1=1'
            params = []
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            if agent_id:
                query += ' AND agent_id = ?'
                params.append(agent_id)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            activities = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return activities
            
        except Exception as e:
            logger.error(f"Error getting agent activity: {e}")
            return []
    
    def get_meetings(self, session_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get meetings with optional filtering."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM meetings WHERE 1=1'
            params = []
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            meetings = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return meetings
            
        except Exception as e:
            logger.error(f"Error getting meetings: {e}")
            return []
    
    def get_metrics(self, session_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get system metrics with optional filtering."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM metrics WHERE 1=1'
            params = []
            
            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)
            else:
                query += ' AND timestamp > datetime("now", "-{} hours")'
                query = query.format(hours)
            
            query += ' ORDER BY timestamp DESC LIMIT 1000'
            
            cursor.execute(query, params)
            metrics = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of session statistics."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get session info
            cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            session = cursor.fetchone()
            if not session:
                conn.close()
                return {}
            
            session = dict(session)
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM chat_logs WHERE session_id = ?', (session_id,))
            chat_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM agent_activity WHERE session_id = ?', (session_id,))
            activity_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM meetings WHERE session_id = ?', (session_id,))
            meeting_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM metrics WHERE session_id = ?', (session_id,))
            metric_count = cursor.fetchone()[0]
            
            # Get unique agents
            cursor.execute('SELECT DISTINCT agent_id FROM agent_activity WHERE session_id = ?', (session_id,))
            unique_agents = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'session_id': session_id,
                'research_question': session.get('research_question', ''),
                'status': session.get('status', 'unknown'),
                'created_at': session.get('created_at'),
                'updated_at': session.get('updated_at'),
                'chat_logs_count': chat_count,
                'agent_activity_count': activity_count,
                'meetings_count': meeting_count,
                'metrics_count': metric_count,
                'unique_agents': unique_agents,
                'agent_count': len(unique_agents)
            }
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            return {}
    
    # SESSION CONTINUITY METHODS
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active (non-completed) sessions."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, created_at, updated_at, status, research_question
                FROM sessions
                WHERE status IN ('pending', 'running', 'paused')
                ORDER BY updated_at DESC
            ''')
            
            sessions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return sessions
            
        except Exception as e:
            logger.error(f"Error getting active sessions: {e}")
            return []
    
    def update_session_status(self, session_id: str, status: str) -> bool:
        """Update session status."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (status, session_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated session {session_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating session status: {e}")
            return False
    
    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Clean up old completed sessions."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old completed sessions and their related data
            cursor.execute('''
                DELETE FROM sessions 
                WHERE status = 'completed' AND created_at < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old sessions")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0
    
    def backup_database(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the database."""
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}.db"
        
        backup_path = self.backup_dir / backup_name
        
        # Create backup
        shutil.copy2(self.db_path, backup_path)
        
        # Create checksum for integrity verification
        checksum = self._calculate_file_checksum(backup_path)
        
        # Store backup metadata
        backup_metadata = {
            'backup_name': backup_name,
            'created_at': datetime.now().isoformat(),
            'original_size': self.db_path.stat().st_size,
            'backup_size': backup_path.stat().st_size,
            'checksum': checksum
        }
        
        metadata_path = backup_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(backup_metadata, f, indent=2)
        
        logger.info(f"Database backup created: {backup_path}")
        return str(backup_path)
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        # Verify backup integrity
        metadata_path = backup_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_checksum = self._calculate_file_checksum(backup_path)
            if current_checksum != metadata['checksum']:
                logger.error("Backup file integrity check failed")
                return False
        
        # Create backup of current database before restore
        current_backup = self.backup_database("pre_restore_backup.db")
        
        try:
            # Restore database
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore database: {e}")
            # Restore from pre-restore backup
            shutil.copy2(current_backup, self.db_path)
            return False
    
    def export_data(self, session_id: Optional[str] = None, format: str = 'json') -> str:
        """Export data to specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = self.archive_dir / f"export_{timestamp}.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Export sessions
            if session_id:
                cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
            else:
                cursor.execute('SELECT * FROM sessions')
            
            sessions = cursor.fetchall()
            sessions_data = [dict(row) for row in sessions]
            
            # Export related data for each session
            for session in sessions_data:
                session_id = session['id']
                
                # Export chat logs
                cursor.execute('SELECT * FROM chat_logs WHERE session_id = ?', (session_id,))
                chat_logs = [dict(row) for row in cursor.fetchall()]
                
                # Export agent activity
                cursor.execute('SELECT * FROM agent_activity WHERE session_id = ?', (session_id,))
                agent_activity = [dict(row) for row in cursor.fetchall()]
                
                # Export meetings
                cursor.execute('SELECT * FROM meetings WHERE session_id = ?', (session_id,))
                meetings = [dict(row) for row in cursor.fetchall()]
                
                # Export metrics
                cursor.execute('SELECT * FROM metrics WHERE session_id = ?', (session_id,))
                metrics = [dict(row) for row in cursor.fetchall()]
                
                # Create session export
                session_export = {
                    'session': session,
                    'chat_logs': chat_logs,
                    'agent_activity': agent_activity,
                    'meetings': meetings,
                    'metrics': metrics
                }
                
                # Add to zip
                zipf.writestr(f"sessions/{session_id}.json", json.dumps(session_export, indent=2))
            
            conn.close()
            
            # Add metadata
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'session_count': len(sessions),
                'format': format,
                'version': '1.0'
            }
            zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        logger.info(f"Data exported to: {export_path}")
        return str(export_path)
    
    def import_data(self, import_path: str) -> bool:
        """Import data from export file."""
        import_path = Path(import_path)
        
        if not import_path.exists():
            logger.error(f"Import file not found: {import_path}")
            return False
        
        try:
            with zipfile.ZipFile(import_path, 'r') as zipf:
                # Read metadata
                metadata = json.loads(zipf.read('metadata.json'))
                
                conn = self.get_db_connection()
                cursor = conn.cursor()
                
                # Import sessions
                for filename in zipf.namelist():
                    if filename.startswith('sessions/') and filename.endswith('.json'):
                        session_data = json.loads(zipf.read(filename))
                        
                        # Import session
                        session = session_data['session']
                        cursor.execute('''
                            INSERT OR REPLACE INTO sessions 
                            (id, created_at, updated_at, status, research_question, config, results, logs, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            session['id'], session['created_at'], session['updated_at'],
                            session['status'], session['research_question'], session['config'],
                            session['results'], session['logs'], session.get('metadata', '{}')
                        ))
                        
                        # Import related data
                        for chat_log in session_data.get('chat_logs', []):
                            cursor.execute('''
                                INSERT INTO chat_logs 
                                (session_id, timestamp, log_type, author, message, metadata)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (
                                chat_log['session_id'], chat_log['timestamp'],
                                chat_log['log_type'], chat_log['author'],
                                chat_log['message'], chat_log.get('metadata', '{}')
                            ))
                        
                        # Import other related data...
                        # (Similar pattern for agent_activity, meetings, metrics)
                
                conn.commit()
                conn.close()
                
                logger.info(f"Data imported from: {import_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and return results."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        integrity_results = {
            'timestamp': datetime.now().isoformat(),
            'tables': {},
            'overall_status': 'valid'
        }
        
        # Check each table
        tables = ['sessions', 'metrics', 'agent_performance', 'chat_logs', 'agent_activity', 'meetings']
        
        for table in tables:
            try:
                # Get record count
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                record_count = cursor.fetchone()[0]
                
                # Calculate checksum (simplified - in practice you'd want more sophisticated checksums)
                cursor.execute(f'SELECT * FROM {table} ORDER BY rowid')
                rows = cursor.fetchall()
                checksum = hashlib.md5(str(rows).encode()).hexdigest()
                
                table_status = {
                    'record_count': record_count,
                    'checksum': checksum,
                    'status': 'valid'
                }
                
                # Store integrity check
                cursor.execute('''
                    INSERT OR REPLACE INTO data_integrity 
                    (table_name, record_count, last_check, checksum, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (table, record_count, datetime.now().isoformat(), checksum, 'valid'))
                
                integrity_results['tables'][table] = table_status
                
            except Exception as e:
                logger.error(f"Error validating table {table}: {e}")
                integrity_results['tables'][table] = {
                    'status': 'error',
                    'error': str(e)
                }
                integrity_results['overall_status'] = 'error'
        
        conn.commit()
        conn.close()
        
        logger.info("Data integrity validation completed")
        return integrity_results
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data and archives."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        cleanup_results = {
            'sessions_removed': 0,
            'backups_removed': 0,
            'archives_removed': 0
        }
        
        # Clean up old sessions
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM sessions 
            WHERE created_at < ?
        ''', (cutoff_date.isoformat(),))
        
        cleanup_results['sessions_removed'] = cursor.rowcount
        
        # Clean up old backups
        for backup_file in self.backup_dir.glob('backup_*.db'):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                cleanup_results['backups_removed'] += 1
        
        # Clean up old archives
        for archive_file in self.archive_dir.glob('export_*.zip'):
            if archive_file.stat().st_mtime < cutoff_date.timestamp():
                archive_file.unlink()
                cleanup_results['archives_removed'] += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Data cleanup completed: {cleanup_results}")
        return cleanup_results
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _start_backup_thread(self):
        """Start automatic backup thread."""
        def backup_worker():
            while True:
                try:
                    # Create daily backup
                    self.backup_database()
                    time.sleep(24 * 60 * 60)  # 24 hours
                except Exception as e:
                    logger.error(f"Backup thread error: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        backup_thread = threading.Thread(target=backup_worker, daemon=True)
        backup_thread.start()
        logger.info("Automatic backup thread started")
    
    def get_data_directory_info(self) -> Dict[str, Any]:
        """Get information about data directory usage."""
        info = {
            'base_directory': str(self.base_dir),
            'data_directory': str(self.data_dir),
            'backup_directory': str(self.backup_dir),
            'archive_directory': str(self.archive_dir),
            'config_directory': str(self.config_dir),
            'logs_directory': str(self.logs_dir),
            'database_path': str(self.db_path),
            'vector_database_path': str(self.vector_db_path),
            'config_path': str(self.config_path),
            'sizes': {},
            'backup_count': 0,
            'archive_count': 0
        }
        
        # Calculate directory sizes
        for directory_name, directory_path in [
            ('data', self.data_dir),
            ('backup', self.backup_dir),
            ('archive', self.archive_dir),
            ('config', self.config_dir),
            ('logs', self.logs_dir)
        ]:
            total_size = 0
            file_count = 0
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            info['sizes'][directory_name] = {
                'size_bytes': total_size,
                'size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count
            }
        
        # Count backups and archives
        info['backup_count'] = len(list(self.backup_dir.glob('backup_*.db')))
        info['archive_count'] = len(list(self.archive_dir.glob('export_*.zip')))
        
        return info 