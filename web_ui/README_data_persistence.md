# Data Persistence System

The AI Research Lab desktop app now includes a comprehensive data persistence system that ensures all data is stored locally and persists between sessions.

## Overview

The data persistence system provides:

- **Local Data Storage**: All data is stored in the user's home directory
- **Automatic Backups**: Daily automatic backups with integrity verification
- **Data Export/Import**: Full data export and import capabilities
- **Data Integrity**: Validation and integrity checks
- **Data Migration**: Automatic migration from old data structure
- **Data Cleanup**: Automatic cleanup of old data and archives

## Data Directory Structure

All data is stored in `~/.ai_research_lab/` with the following structure:

```
~/.ai_research_lab/
├── data/                    # Main data directory
│   ├── research_sessions.db # SQLite database
│   ├── vector_memory.db     # Vector database for memory
│   └── output/              # Research output files
├── backups/                 # Database backups
│   ├── backup_20241201_143022.db
│   └── backup_20241201_143022.json
├── archives/                # Data exports
│   ├── export_20241201_143022.zip
│   └── export_20241201_143022.json
├── config/                  # Configuration files
│   └── config.json
└── logs/                    # Application logs
    ├── migration_log_170143022.json
    └── app.log
```

## Features

### 1. Automatic Data Management

- **Database Initialization**: Automatically creates and initializes SQLite database with proper schema
- **Directory Creation**: Creates all necessary directories on first run
- **Configuration Management**: Stores and manages application configuration

### 2. Backup and Recovery

- **Automatic Backups**: Daily automatic backups with checksum verification
- **Manual Backups**: Create backups on demand via API
- **Backup Restoration**: Restore from any backup with integrity verification
- **Backup Metadata**: Each backup includes metadata for verification

### 3. Data Export and Import

- **Full Data Export**: Export all data as compressed ZIP archive
- **Session Export**: Export specific session data
- **Data Import**: Import data from export archives
- **Format Support**: JSON format with metadata

### 4. Data Integrity

- **Integrity Validation**: Regular validation of database integrity
- **Checksum Verification**: MD5 checksums for backup verification
- **Error Detection**: Automatic detection and reporting of data issues
- **Recovery**: Automatic recovery from corrupted data

### 5. Data Cleanup

- **Automatic Cleanup**: Remove old sessions, backups, and archives
- **Configurable Retention**: Configurable retention periods
- **Safe Cleanup**: Always creates backups before cleanup
- **Cleanup Logging**: Detailed logging of cleanup operations

### 6. Data Migration

- **Automatic Migration**: Migrate from old data structure
- **Backup Creation**: Creates backups before migration
- **Verification**: Verifies migration success
- **Rollback**: Ability to rollback failed migrations

## API Endpoints

The web UI provides the following data management endpoints:

### Backup and Recovery
- `POST /api/data/backup` - Create a database backup
- `POST /api/data/restore` - Restore database from backup

### Data Export/Import
- `POST /api/data/export` - Export data to archive
- `POST /api/data/import` - Import data from archive

### Data Management
- `GET /api/data/integrity` - Validate data integrity
- `POST /api/data/cleanup` - Clean up old data
- `GET /api/data/info` - Get data directory information
- `POST /api/data/migrate` - Migrate existing data

## Usage

### Starting the Application

The data persistence system is automatically initialized when the application starts:

```bash
cd web_ui
python app.py
```

The system will:
1. Create the data directory structure
2. Initialize the database
3. Migrate existing data (if any)
4. Start automatic backup thread

### Testing the System

Run the test script to verify the data persistence system:

```bash
cd web_ui
python test_data_persistence.py
```

### Manual Data Migration

If you have existing data in the web_ui directory, it will be automatically migrated. You can also trigger migration manually:

```bash
curl -X POST http://localhost:5000/api/data/migrate
```

### Creating Backups

Create a manual backup:

```bash
curl -X POST http://localhost:5000/api/data/backup
```

### Exporting Data

Export all data:

```bash
curl -X POST http://localhost:5000/api/data/export
```

Export specific session:

```bash
curl -X POST http://localhost:5000/api/data/export \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your_session_id"}'
```

### Checking Data Integrity

Validate data integrity:

```bash
curl http://localhost:5000/api/data/integrity
```

### Getting Data Information

Get data directory information:

```bash
curl http://localhost:5000/api/data/info
```

## Configuration

The data persistence system uses the following configuration:

### Data Directory
- **Default**: `~/.ai_research_lab/`
- **Configurable**: Set `base_dir` parameter in DataManager constructor

### Backup Settings
- **Frequency**: Daily automatic backups
- **Retention**: Configurable (default: 30 days)
- **Location**: `~/.ai_research_lab/backups/`

### Cleanup Settings
- **Session Retention**: 30 days (configurable)
- **Backup Retention**: 30 days (configurable)
- **Archive Retention**: 30 days (configurable)

## Database Schema

The SQLite database includes the following tables:

### Sessions Table
- `id` - Session identifier
- `created_at` - Creation timestamp
- `updated_at` - Last update timestamp
- `status` - Session status
- `research_question` - Research question
- `config` - Session configuration (JSON)
- `results` - Session results (JSON)
- `logs` - Session logs (JSON)
- `metadata` - Additional metadata (JSON)

### Metrics Table
- `id` - Metric identifier
- `timestamp` - Metric timestamp
- `cpu_usage` - CPU usage percentage
- `memory_usage` - Memory usage percentage
- `active_agents` - Number of active agents
- `session_id` - Associated session

### Chat Logs Table
- `id` - Log identifier
- `session_id` - Associated session
- `timestamp` - Log timestamp
- `log_type` - Log type (thought, choice, communication, tool_call, system)
- `author` - Log author
- `message` - Log message
- `metadata` - Additional metadata (JSON)

### Agent Activity Table
- `id` - Activity identifier
- `session_id` - Associated session
- `agent_id` - Agent identifier
- `timestamp` - Activity timestamp
- `activity_type` - Activity type (thinking, speaking, tool_use, meeting, idle)
- `message` - Activity message
- `status` - Activity status
- `metadata` - Additional metadata (JSON)

### Meetings Table
- `id` - Meeting identifier
- `session_id` - Associated session
- `meeting_id` - Meeting identifier
- `timestamp` - Meeting timestamp
- `participants` - Meeting participants (JSON)
- `topic` - Meeting topic
- `agenda` - Meeting agenda (JSON)
- `transcript` - Meeting transcript
- `outcomes` - Meeting outcomes (JSON)
- `metadata` - Additional metadata (JSON)

## Troubleshooting

### Common Issues

1. **Permission Errors**
   - Ensure the application has write permissions to the home directory
   - Check if the data directory exists and is writable

2. **Database Corruption**
   - Use the integrity validation endpoint to check for corruption
   - Restore from a recent backup if corruption is detected

3. **Migration Failures**
   - Check the migration logs in the logs directory
   - Ensure sufficient disk space for migration
   - Verify file permissions

4. **Backup Failures**
   - Check available disk space
   - Verify write permissions to backup directory
   - Check for file locks

### Log Files

The system creates detailed logs in the `~/.ai_research_lab/logs/` directory:

- `migration_log_*.json` - Migration operation logs
- `app.log` - Application logs (if configured)

### Data Recovery

If data is lost or corrupted:

1. **Check Backups**: Look in `~/.ai_research_lab/backups/`
2. **Restore from Backup**: Use the restore endpoint
3. **Check Archives**: Look in `~/.ai_research_lab/archives/`
4. **Import from Archive**: Use the import endpoint

## Security Considerations

- **Local Storage**: All data is stored locally on the user's machine
- **No Network Transmission**: Data is not transmitted over the network
- **File Permissions**: Data files use standard file system permissions
- **Backup Security**: Backups inherit file system security

## Performance Considerations

- **Database Optimization**: SQLite database is optimized for local use
- **Backup Compression**: Backups are compressed to save space
- **Cleanup Automation**: Automatic cleanup prevents disk space issues
- **Indexing**: Database tables are properly indexed for performance

## Future Enhancements

- **Encryption**: Optional encryption for sensitive data
- **Cloud Backup**: Optional cloud backup integration
- **Data Compression**: Advanced compression for large datasets
- **Incremental Backups**: Incremental backup support
- **Data Analytics**: Built-in data analytics and reporting 