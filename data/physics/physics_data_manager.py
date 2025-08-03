"""
Physics Data Manager for AI Research Lab Framework.

Manages physics-specific data operations including loading, processing, and storage
of experimental data, simulation results, and theoretical calculations.
"""

import os
import json
import logging
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil

# Try to import optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

@dataclass
class PhysicsDataset:
    """Represents a physics dataset with metadata."""
    id: str
    name: str
    data_type: str  # experimental, simulation, theoretical, reference
    domain: str     # particle, condensed_matter, astrophysics, etc.
    source: str
    format: str     # json, hdf5, csv, txt, binary
    created_at: datetime
    metadata: Dict[str, Any]
    file_path: Optional[str] = None
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result

class PhysicsDataManager:
    """Manages physics-specific data operations."""
    
    def __init__(self, config: Dict[str, Any], data_dir: Optional[str] = None):
        """
        Initialize the Physics Data Manager.
        
        Args:
            config: Configuration dictionary
            data_dir: Directory for storing physics data files
        """
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".ai_research_lab" / "physics_data"
        self.cache_dir = self.data_dir / "cache"
        self.temp_dir = self.data_dir / "temp"
        
        # Database path for physics data metadata
        self.db_path = self.data_dir / "physics_data.db"
        
        # Initialize directories
        self._create_directories()
        
        # Initialize database
        self._init_database()
        
        # Data registry for loaded datasets
        self._data_registry: Dict[str, PhysicsDataset] = {}
        
        # Supported formats
        self.supported_formats = {
            'json': self._load_json,
            'csv': self._load_csv,
            'txt': self._load_text,
            'hdf5': self._load_hdf5 if HDF5_AVAILABLE else None,
            'h5': self._load_hdf5 if HDF5_AVAILABLE else None,
            'binary': self._load_binary
        }
        
        logger.info(f"PhysicsDataManager initialized with data directory: {self.data_dir}")
    
    def _create_directories(self) -> None:
        """Create necessary directory structure."""
        directories = [
            self.data_dir,
            self.cache_dir,
            self.temp_dir,
            self.data_dir / "experimental",
            self.data_dir / "simulation",
            self.data_dir / "theoretical",
            self.data_dir / "reference"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Created physics data directory structure")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for physics data metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create physics datasets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS physics_datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                format TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Create data lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                parent_dataset_id TEXT,
                operation TEXT NOT NULL,
                parameters TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dataset_id) REFERENCES physics_datasets(id),
                FOREIGN KEY (parent_dataset_id) REFERENCES physics_datasets(id)
            )
        ''')
        
        # Create data access log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                error_message TEXT,
                FOREIGN KEY (dataset_id) REFERENCES physics_datasets(id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_type ON physics_datasets(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_domain ON physics_datasets(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_created ON physics_datasets(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_dataset ON data_lineage(dataset_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_log_dataset ON data_access_log(dataset_id)')
        
        conn.commit()
        conn.close()
        
        logger.info("Physics data database initialized")
    
    def get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_physics_data(self, source: str, format: str, **kwargs) -> Optional[PhysicsDataset]:
        """
        Load physics data from various sources.
        
        Args:
            source: Data source (file path, URL, database identifier)
            format: Data format (json, csv, hdf5, etc.)
            **kwargs: Additional parameters for data loading
        
        Returns:
            PhysicsDataset object or None if loading fails
        """
        try:
            # Generate unique ID for the dataset
            dataset_id = self._generate_dataset_id(source, format)
            
            # Check if already loaded
            if dataset_id in self._data_registry:
                logger.info(f"Dataset {dataset_id} already loaded from cache")
                return self._data_registry[dataset_id]
            
            # Validate format
            if format.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported format: {format}")
            
            loader = self.supported_formats[format.lower()]
            if loader is None:
                raise ValueError(f"Format {format} not available (missing dependencies)")
            
            # Load data
            data, metadata = loader(source, **kwargs)
            
            # Create dataset object
            dataset = PhysicsDataset(
                id=dataset_id,
                name=kwargs.get('name', f"dataset_{dataset_id[:8]}"),
                data_type=kwargs.get('data_type', 'experimental'),
                domain=kwargs.get('domain', 'general'),
                source=source,
                format=format.lower(),
                created_at=datetime.now(),
                metadata=metadata,
                file_path=source if Path(source).exists() else None,
                data=data
            )
            
            # Register dataset
            self._data_registry[dataset_id] = dataset
            
            # Persist metadata to database
            self._persist_dataset_metadata(dataset)
            
            # Log access
            self._log_data_access(dataset_id, 'load', 'success')
            
            logger.info(f"Successfully loaded physics dataset: {dataset.name}")
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to load physics data from {source}: {str(e)}"
            logger.error(error_msg)
            
            # Log failed access
            if 'dataset_id' in locals():
                self._log_data_access(dataset_id, 'load', 'error', error_msg)
            
            return None
    
    def save_physics_data(self, dataset: PhysicsDataset, file_path: Optional[str] = None) -> bool:
        """
        Save physics dataset to file.
        
        Args:
            dataset: PhysicsDataset to save
            file_path: Optional custom file path
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path is None:
                # Generate file path based on dataset properties
                subdir = self.data_dir / dataset.data_type
                subdir.mkdir(exist_ok=True)
                file_path = subdir / f"{dataset.name}_{dataset.id[:8]}.{dataset.format}"
            
            file_path = Path(file_path)
            
            # Save data based on format
            if dataset.format == 'json':
                self._save_json(dataset.data, file_path, dataset.metadata)
            elif dataset.format == 'csv' and PANDAS_AVAILABLE:
                self._save_csv(dataset.data, file_path)
            elif dataset.format in ['hdf5', 'h5'] and HDF5_AVAILABLE:
                self._save_hdf5(dataset.data, file_path, dataset.metadata)
            else:
                raise ValueError(f"Saving not supported for format: {dataset.format}")
            
            # Update dataset file path
            dataset.file_path = str(file_path)
            
            # Update database
            self._update_dataset_metadata(dataset)
            
            # Log access
            self._log_data_access(dataset.id, 'save', 'success')
            
            logger.info(f"Successfully saved physics dataset to: {file_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to save physics dataset {dataset.id}: {str(e)}"
            logger.error(error_msg)
            self._log_data_access(dataset.id, 'save', 'error', error_msg)
            return False
    
    def get_dataset(self, dataset_id: str) -> Optional[PhysicsDataset]:
        """Get dataset by ID."""
        # Check memory registry first
        if dataset_id in self._data_registry:
            return self._data_registry[dataset_id]
        
        # Check database
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM physics_datasets WHERE id = ?', (dataset_id,))
            row = cursor.fetchone()
            
            if row:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                dataset = PhysicsDataset(
                    id=row['id'],
                    name=row['name'],
                    data_type=row['data_type'],
                    domain=row['domain'],
                    source=row['source'],
                    format=row['format'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    metadata=metadata,
                    file_path=row['file_path']
                )
                
                # Load data if file exists
                if dataset.file_path and Path(dataset.file_path).exists():
                    loaded_dataset = self.load_physics_data(dataset.file_path, dataset.format)
                    if loaded_dataset:
                        dataset.data = loaded_dataset.data
                
                self._data_registry[dataset_id] = dataset
                conn.close()
                return dataset
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {str(e)}")
            return None
    
    def list_datasets(self, data_type: Optional[str] = None, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available physics datasets.
        
        Args:
            data_type: Filter by data type
            domain: Filter by domain
        
        Returns:
            List of dataset metadata dictionaries
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM physics_datasets WHERE status = "active"'
            params = []
            
            if data_type:
                query += ' AND data_type = ?'
                params.append(data_type)
            
            if domain:
                query += ' AND domain = ?'
                params.append(domain)
            
            query += ' ORDER BY created_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            datasets = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                datasets.append({
                    'id': row['id'],
                    'name': row['name'],
                    'data_type': row['data_type'],
                    'domain': row['domain'],
                    'source': row['source'],
                    'format': row['format'],
                    'created_at': row['created_at'],
                    'file_path': row['file_path'],
                    'metadata': metadata
                })
            
            conn.close()
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            return []
    
    def delete_dataset(self, dataset_id: str, remove_file: bool = False) -> bool:
        """
        Delete a physics dataset.
        
        Args:
            dataset_id: Dataset ID to delete
            remove_file: Whether to remove the physical file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get dataset info
            dataset = self.get_dataset(dataset_id)
            if not dataset:
                logger.warning(f"Dataset {dataset_id} not found")
                return False
            
            # Remove file if requested
            if remove_file and dataset.file_path:
                file_path = Path(dataset.file_path)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed file: {file_path}")
            
            # Mark as deleted in database
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE physics_datasets SET status = "deleted", updated_at = CURRENT_TIMESTAMP WHERE id = ?',
                (dataset_id,)
            )
            
            conn.commit()
            conn.close()
            
            # Remove from memory registry
            if dataset_id in self._data_registry:
                del self._data_registry[dataset_id]
            
            # Log access
            self._log_data_access(dataset_id, 'delete', 'success')
            
            logger.info(f"Successfully deleted dataset: {dataset_id}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete dataset {dataset_id}: {str(e)}"
            logger.error(error_msg)
            self._log_data_access(dataset_id, 'delete', 'error', error_msg)
            return False
    
    # Format-specific loaders
    def _load_json(self, source: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load JSON data."""
        with open(source, 'r') as f:
            data = json.load(f)
        
        metadata = kwargs.copy()
        metadata.update({
            'file_size': os.path.getsize(source),
            'encoding': kwargs.get('encoding', 'utf-8')
        })
        
        return data, metadata
    
    def _load_csv(self, source: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load CSV data."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV support")
        
        data = pd.read_csv(source, **kwargs)
        
        metadata = kwargs.copy()
        metadata.update({
            'file_size': os.path.getsize(source),
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
        })
        
        return data, metadata
    
    def _load_text(self, source: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load text data."""
        encoding = kwargs.get('encoding', 'utf-8')
        
        with open(source, 'r', encoding=encoding) as f:
            data = f.read()
        
        metadata = kwargs.copy()
        metadata.update({
            'file_size': os.path.getsize(source),
            'encoding': encoding,
            'line_count': len(data.splitlines())
        })
        
        return data, metadata
    
    def _load_hdf5(self, source: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load HDF5 data."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")
        
        data = {}
        metadata = kwargs.copy()
        
        with h5py.File(source, 'r') as f:
            # Load all datasets
            def visit_func(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[...]
            
            f.visititems(visit_func)
            
            # Extract metadata
            metadata.update({
                'file_size': os.path.getsize(source),
                'datasets': list(data.keys()),
                'hdf5_attrs': dict(f.attrs) if f.attrs else {}
            })
        
        return data, metadata
    
    def _load_binary(self, source: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Load binary data."""
        with open(source, 'rb') as f:
            data = f.read()
        
        metadata = kwargs.copy()
        metadata.update({
            'file_size': os.path.getsize(source),
            'data_type': 'binary'
        })
        
        return data, metadata
    
    # Format-specific savers
    def _save_json(self, data: Any, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save data as JSON."""
        output = {
            'data': data,
            'metadata': metadata
        }
        
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
    
    def _save_csv(self, data: Any, file_path: Path) -> None:
        """Save data as CSV."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV support")
        
        if hasattr(data, 'to_csv'):
            data.to_csv(file_path, index=False)
        else:
            # Convert to DataFrame if possible
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
    
    def _save_hdf5(self, data: Any, file_path: Path, metadata: Dict[str, Any]) -> None:
        """Save data as HDF5."""
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 support")
        
        with h5py.File(file_path, 'w') as f:
            # Save metadata as attributes
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except (TypeError, ValueError):
                    f.attrs[key] = str(value)
            
            # Save data
            if isinstance(data, dict):
                for key, value in data.items():
                    try:
                        f.create_dataset(key, data=value)
                    except (TypeError, ValueError):
                        # Convert to string if can't be saved directly
                        f.create_dataset(key, data=str(value))
            else:
                f.create_dataset('data', data=data)
    
    # Helper methods
    def _generate_dataset_id(self, source: str, format: str) -> str:
        """Generate unique dataset ID."""
        import hashlib
        unique_string = f"{source}_{format}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _persist_dataset_metadata(self, dataset: PhysicsDataset) -> None:
        """Persist dataset metadata to database."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO physics_datasets
            (id, name, data_type, domain, source, format, created_at, file_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset.id,
            dataset.name,
            dataset.data_type,
            dataset.domain,
            dataset.source,
            dataset.format,
            dataset.created_at.isoformat(),
            dataset.file_path,
            json.dumps(dataset.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _update_dataset_metadata(self, dataset: PhysicsDataset) -> None:
        """Update dataset metadata in database."""
        conn = self.get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE physics_datasets
            SET name = ?, file_path = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            dataset.name,
            dataset.file_path,
            json.dumps(dataset.metadata),
            dataset.id
        ))
        
        conn.commit()
        conn.close()
    
    def _log_data_access(self, dataset_id: str, operation: str, status: str, error_message: Optional[str] = None) -> None:
        """Log data access operation."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_access_log (dataset_id, operation, status, error_message)
                VALUES (?, ?, ?, ?)
            ''', (dataset_id, operation, status, error_message))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log data access: {str(e)}")
    
    async def load_physics_data_async(self, source: str, format: str, **kwargs) -> Optional[PhysicsDataset]:
        """Async version of load_physics_data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_physics_data, source, format, **kwargs)
    
    async def save_physics_data_async(self, dataset: PhysicsDataset, file_path: Optional[str] = None) -> bool:
        """Async version of save_physics_data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.save_physics_data, dataset, file_path)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored physics data."""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Count by data type
            cursor.execute('''
                SELECT data_type, COUNT(*) as count
                FROM physics_datasets
                WHERE status = 'active'
                GROUP BY data_type
            ''')
            type_counts = dict(cursor.fetchall())
            
            # Count by domain
            cursor.execute('''
                SELECT domain, COUNT(*) as count
                FROM physics_datasets
                WHERE status = 'active'
                GROUP BY domain
            ''')
            domain_counts = dict(cursor.fetchall())
            
            # Count by format
            cursor.execute('''
                SELECT format, COUNT(*) as count
                FROM physics_datasets
                WHERE status = 'active'
                GROUP BY format
            ''')
            format_counts = dict(cursor.fetchall())
            
            # Total count
            cursor.execute('SELECT COUNT(*) FROM physics_datasets WHERE status = "active"')
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_datasets': total_count,
                'by_type': type_counts,
                'by_domain': domain_counts,
                'by_format': format_counts,
                'memory_loaded': len(self._data_registry)
            }
            
        except Exception as e:
            logger.error(f"Failed to get data statistics: {str(e)}")
            return {}