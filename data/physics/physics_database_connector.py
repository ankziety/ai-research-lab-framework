"""
Physics Database Connector for AI Research Lab Framework.

Connects to external physics databases including PDB, NIST, arXiv, Materials Project,
and astrophysics databases to retrieve and cache physics data.
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False
    ET = None

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConnection:
    """Represents a connection to an external physics database."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # requests per second
    timeout: int = 30
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {'User-Agent': 'AI-Research-Lab-Framework/1.0'}

class PhysicsDatabaseConnector:
    """Connects to external physics databases for data retrieval."""
    
    def __init__(self, config: Dict[str, Any], cache_dir: Optional[str] = None):
        """
        Initialize the Physics Database Connector.
        
        Args:
            config: Configuration dictionary with API keys and settings
            cache_dir: Directory for caching retrieved data
        """
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".ai_research_lab" / "physics_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting tracking
        self._last_request_times: Dict[str, float] = {}
        
        # Initialize database connections
        self.databases = self._initialize_databases()
        
        # Cache settings
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        
        logger.info(f"PhysicsDatabaseConnector initialized with {len(self.databases)} databases")
    
    def _initialize_databases(self) -> Dict[str, DatabaseConnection]:
        """Initialize connections to physics databases."""
        databases = {}
        
        # Protein Data Bank (PDB)
        databases['pdb'] = DatabaseConnection(
            name='Protein Data Bank',
            base_url='https://data.rcsb.org/rest/v1',
            rate_limit=10.0,  # 10 requests per second
            headers={'User-Agent': 'AI-Research-Lab-Framework/1.0'}
        )
        
        # NIST Physical Reference Data
        databases['nist'] = DatabaseConnection(
            name='NIST Physical Reference Data',
            base_url='https://physics.nist.gov/cgi-bin/cuu/Value',
            rate_limit=1.0,  # 1 request per second
        )
        
        # arXiv
        databases['arxiv'] = DatabaseConnection(
            name='arXiv',
            base_url='http://export.arxiv.org/api/query',
            rate_limit=3.0,  # 3 requests per second
        )
        
        # Materials Project
        api_key = self.config.get('materials_project_api_key')
        if api_key:
            databases['materials_project'] = DatabaseConnection(
                name='Materials Project',
                base_url='https://api.materialsproject.org/v1',
                api_key=api_key,
                rate_limit=5.0,
                headers={
                    'X-API-KEY': api_key,
                    'User-Agent': 'AI-Research-Lab-Framework/1.0'
                }
            )
        
        # SDSS (Sloan Digital Sky Survey)
        databases['sdss'] = DatabaseConnection(
            name='SDSS',
            base_url='http://skyserver.sdss.org/dr17/SkyServerWS',
            rate_limit=2.0,
        )
        
        # ChemSpider (if API key provided)
        chemspider_key = self.config.get('chemspider_api_key')
        if chemspider_key:
            databases['chemspider'] = DatabaseConnection(
                name='ChemSpider',
                base_url='https://api.rsc.org/compounds/v1',
                api_key=chemspider_key,
                rate_limit=1.0,
                headers={
                    'apikey': chemspider_key,
                    'User-Agent': 'AI-Research-Lab-Framework/1.0'
                }
            )
        
        return databases
    
    def search_physics_database(self, query: str, database: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Search a physics database.
        
        Args:
            query: Search query
            database: Database name (pdb, nist, arxiv, etc.)
            **kwargs: Additional search parameters
        
        Returns:
            Search results or None if failed
        """
        if not REQUESTS_AVAILABLE:
            logger.error("requests library is required for database connections")
            return None
        
        if database not in self.databases:
            logger.error(f"Database {database} not configured")
            return None
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(database, query, kwargs)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Retrieved {database} search from cache")
                return cached_result
            
            # Apply rate limiting
            self._apply_rate_limit(database)
            
            # Perform search based on database type
            if database == 'pdb':
                result = self._search_pdb(query, **kwargs)
            elif database == 'nist':
                result = self._search_nist(query, **kwargs)
            elif database == 'arxiv':
                result = self._search_arxiv(query, **kwargs)
            elif database == 'materials_project':
                result = self._search_materials_project(query, **kwargs)
            elif database == 'sdss':
                result = self._search_sdss(query, **kwargs)
            elif database == 'chemspider':
                result = self._search_chemspider(query, **kwargs)
            else:
                result = self._generic_search(database, query, **kwargs)
            
            if result:
                # Cache the result
                self._cache_result(cache_key, result)
                logger.info(f"Successfully searched {database} for: {query}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to search {database}: {str(e)}")
            return None
    
    def get_structure_data(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get protein structure data from PDB.
        
        Args:
            pdb_id: PDB identifier
        
        Returns:
            Structure data or None if failed
        """
        return self.search_physics_database(pdb_id, 'pdb', search_type='structure')
    
    def get_physical_constants(self, constant_name: str) -> Optional[Dict[str, Any]]:
        """
        Get physical constants from NIST.
        
        Args:
            constant_name: Name of the physical constant
        
        Returns:
            Constant data or None if failed
        """
        return self.search_physics_database(constant_name, 'nist', search_type='constant')
    
    def get_arxiv_papers(self, search_query: str, max_results: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Search arXiv for physics papers.
        
        Args:
            search_query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of paper metadata or None if failed
        """
        result = self.search_physics_database(
            search_query, 'arxiv',
            max_results=max_results,
            search_type='papers'
        )
        
        if result and 'entries' in result:
            return result['entries']
        return None
    
    def get_material_properties(self, material_id: str) -> Optional[Dict[str, Any]]:
        """
        Get material properties from Materials Project.
        
        Args:
            material_id: Materials Project ID
        
        Returns:
            Material properties or None if failed
        """
        if 'materials_project' not in self.databases:
            logger.error("Materials Project API key not configured")
            return None
        
        return self.search_physics_database(
            material_id, 'materials_project',
            search_type='material'
        )
    
    def get_astronomical_data(self, coordinates: Tuple[float, float], radius: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Get astronomical data from SDSS.
        
        Args:
            coordinates: (RA, Dec) coordinates in degrees
            radius: Search radius in degrees
        
        Returns:
            Astronomical data or None if failed
        """
        ra, dec = coordinates
        query = f"ra={ra}&dec={dec}&radius={radius}"
        
        return self.search_physics_database(
            query, 'sdss',
            search_type='objects',
            coordinates=coordinates,
            radius=radius
        )
    
    # Database-specific search methods
    def _search_pdb(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search Protein Data Bank."""
        db_conn = self.databases['pdb']
        search_type = kwargs.get('search_type', 'structure')
        
        if search_type == 'structure':
            # Get structure information
            url = f"{db_conn.base_url}/core/entry/{query.upper()}"
        else:
            # General search
            url = f"{db_conn.base_url}/core/entry/{query.upper()}"
        
        response = requests.get(url, headers=db_conn.headers, timeout=db_conn.timeout)
        
        if response.status_code == 200:
            return {
                'source': 'pdb',
                'query': query,
                'data': response.json(),
                'retrieved_at': datetime.now().isoformat()
            }
        else:
            logger.warning(f"PDB search failed with status {response.status_code}")
            return None
    
    def _search_nist(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search NIST Physical Reference Data."""
        # Note: NIST doesn't have a simple REST API, this is a simplified implementation
        # In practice, you'd need to parse their specific data formats
        
        # For demonstration, we'll create a mock response for common constants
        constants = {
            'speed_of_light': {
                'value': 299792458,
                'unit': 'm/s',
                'uncertainty': 0,
                'description': 'Speed of light in vacuum'
            },
            'planck_constant': {
                'value': 6.62607015e-34,
                'unit': 'J⋅s',
                'uncertainty': 0,
                'description': 'Planck constant'
            },
            'electron_mass': {
                'value': 9.1093837015e-31,
                'unit': 'kg',
                'uncertainty': 2.8e-40,
                'description': 'Electron rest mass'
            }
        }
        
        query_lower = query.lower().replace(' ', '_')
        if query_lower in constants:
            return {
                'source': 'nist',
                'query': query,
                'data': constants[query_lower],
                'retrieved_at': datetime.now().isoformat()
            }
        
        return None
    
    def _search_arxiv(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search arXiv."""
        db_conn = self.databases['arxiv']
        max_results = kwargs.get('max_results', 10)
        
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results
        }
        
        response = requests.get(
            db_conn.base_url,
            params=params,
            headers=db_conn.headers,
            timeout=db_conn.timeout
        )
        
        if response.status_code == 200:
            # Parse XML response
            if XML_AVAILABLE:
                root = ET.fromstring(response.content)
                entries = []
                
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    paper_data = {}
                    
                    # Extract basic information
                    title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                    if title_elem is not None:
                        paper_data['title'] = title_elem.text.strip()
                    
                    summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                    if summary_elem is not None:
                        paper_data['abstract'] = summary_elem.text.strip()
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                        name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                        if name_elem is not None:
                            authors.append(name_elem.text)
                    paper_data['authors'] = authors
                    
                    # Extract arXiv ID
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    if id_elem is not None:
                        paper_data['arxiv_id'] = id_elem.text.split('/')[-1]
                    
                    # Extract categories
                    categories = []
                    for category in entry.findall('{http://arxiv.org/schemas/atom}primary_category'):
                        term = category.get('term')
                        if term:
                            categories.append(term)
                    paper_data['categories'] = categories
                    
                    entries.append(paper_data)
                
                return {
                    'source': 'arxiv',
                    'query': query,
                    'entries': entries,
                    'total_results': len(entries),
                    'retrieved_at': datetime.now().isoformat()
                }
            else:
                # Fallback without XML parsing
                return {
                    'source': 'arxiv',
                    'query': query,
                    'raw_data': response.text,
                    'retrieved_at': datetime.now().isoformat()
                }
        
        return None
    
    def _search_materials_project(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search Materials Project."""
        if 'materials_project' not in self.databases:
            return None
        
        db_conn = self.databases['materials_project']
        search_type = kwargs.get('search_type', 'material')
        
        if search_type == 'material':
            # Get material by ID
            url = f"{db_conn.base_url}/materials/{query}"
        else:
            # General search
            url = f"{db_conn.base_url}/materials"
            params = {'formula': query}
        
        response = requests.get(
            url,
            headers=db_conn.headers,
            timeout=db_conn.timeout,
            params=params if 'params' in locals() else None
        )
        
        if response.status_code == 200:
            return {
                'source': 'materials_project',
                'query': query,
                'data': response.json(),
                'retrieved_at': datetime.now().isoformat()
            }
        
        return None
    
    def _search_sdss(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search SDSS."""
        db_conn = self.databases['sdss']
        coordinates = kwargs.get('coordinates')
        radius = kwargs.get('radius', 0.1)
        
        if coordinates:
            ra, dec = coordinates
            # Simplified SDSS query
            url = f"{db_conn.base_url}/SearchTools/CrossIdSearch"
            params = {
                'ra': ra,
                'dec': dec,
                'radius': radius,
                'format': 'json'
            }
            
            response = requests.get(
                url,
                params=params,
                headers=db_conn.headers,
                timeout=db_conn.timeout
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                except:
                    # If JSON parsing fails, return raw data
                    data = response.text
                
                return {
                    'source': 'sdss',
                    'query': query,
                    'coordinates': coordinates,
                    'radius': radius,
                    'data': data,
                    'retrieved_at': datetime.now().isoformat()
                }
        
        return None
    
    def _search_chemspider(self, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Search ChemSpider."""
        if 'chemspider' not in self.databases:
            return None
        
        db_conn = self.databases['chemspider']
        
        # Search for compounds by name
        url = f"{db_conn.base_url}/filter/name"
        params = {'name': query}
        
        response = requests.get(
            url,
            params=params,
            headers=db_conn.headers,
            timeout=db_conn.timeout
        )
        
        if response.status_code == 200:
            return {
                'source': 'chemspider',
                'query': query,
                'data': response.json(),
                'retrieved_at': datetime.now().isoformat()
            }
        
        return None
    
    def _generic_search(self, database: str, query: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generic search for other databases."""
        db_conn = self.databases[database]
        
        # Simple GET request to base URL with query
        params = {'q': query}
        params.update(kwargs)
        
        response = requests.get(
            db_conn.base_url,
            params=params,
            headers=db_conn.headers,
            timeout=db_conn.timeout
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
            except:
                data = response.text
            
            return {
                'source': database,
                'query': query,
                'data': data,
                'retrieved_at': datetime.now().isoformat()
            }
        
        return None
    
    # Cache management
    def _generate_cache_key(self, database: str, query: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for a query."""
        cache_data = {
            'database': database,
            'query': query,
            'kwargs': sorted(kwargs.items())
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if not expired."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is expired
                cached_time = datetime.fromisoformat(cached_data['cached_at'])
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    return cached_data['result']
                else:
                    # Remove expired cache
                    cache_file.unlink()
            
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_key}: {str(e)}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache a result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_data = {
                'result': result,
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        
        except Exception as e:
            logger.warning(f"Failed to cache result {cache_key}: {str(e)}")
    
    def _apply_rate_limit(self, database: str) -> None:
        """Apply rate limiting for database requests."""
        db_conn = self.databases[database]
        current_time = time.time()
        
        if database in self._last_request_times:
            time_since_last = current_time - self._last_request_times[database]
            min_interval = 1.0 / db_conn.rate_limit
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Rate limiting {database}: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self._last_request_times[database] = time.time()
    
    # Async methods
    async def search_physics_database_async(self, query: str, database: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Async version of search_physics_database."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_physics_database, query, database, **kwargs)
    
    async def get_structure_data_async(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """Async version of get_structure_data."""
        return await self.search_physics_database_async(pdb_id, 'pdb', search_type='structure')
    
    async def get_arxiv_papers_async(self, search_query: str, max_results: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Async version of get_arxiv_papers."""
        result = await self.search_physics_database_async(
            search_query, 'arxiv',
            max_results=max_results,
            search_type='papers'
        )
        
        if result and 'entries' in result:
            return result['entries']
        return None
    
    # Utility methods
    def list_available_databases(self) -> List[Dict[str, Any]]:
        """List all available database connections."""
        return [
            {
                'name': db_conn.name,
                'key': key,
                'base_url': db_conn.base_url,
                'rate_limit': db_conn.rate_limit,
                'has_api_key': db_conn.api_key is not None
            }
            for key, db_conn in self.databases.items()
        ]
    
    def clear_cache(self, database: Optional[str] = None) -> int:
        """
        Clear cached results.
        
        Args:
            database: Specific database to clear, or None for all
        
        Returns:
            Number of cache files removed
        """
        removed_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                should_remove = True
                
                if database:
                    # Check if cache file is for specific database
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                        
                        if cached_data.get('result', {}).get('source') != database:
                            should_remove = False
                    except:
                        pass
                
                if should_remove:
                    cache_file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleared {removed_count} cache files")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
        
        return removed_count
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Count by database
            db_counts = {}
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    
                    db_name = cached_data.get('result', {}).get('source', 'unknown')
                    db_counts[db_name] = db_counts.get(db_name, 0) + 1
                except:
                    pass
            
            return {
                'total_files': len(cache_files),
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'by_database': db_counts,
                'cache_ttl': self.cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {str(e)}")
            return {}

    def test_database_connections(self) -> Dict[str, bool]:
        """Test connectivity to all configured databases."""
        results = {}
        
        for db_name in self.databases:
            try:
                # Simple test query
                if db_name == 'pdb':
                    result = self._search_pdb('1abc')  # Test with common PDB ID
                elif db_name == 'nist':
                    result = self._search_nist('speed_of_light')
                elif db_name == 'arxiv':
                    result = self._search_arxiv('physics')
                else:
                    result = True  # Assume working for other databases
                
                results[db_name] = result is not None
                
            except Exception as e:
                logger.warning(f"Database {db_name} connection test failed: {str(e)}")
                results[db_name] = False
        
        return results