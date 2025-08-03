"""
Physics Data Export for AI Research Lab Framework.

Provides export and reporting functionality for physics results including
manuscript generation, data archiving, and format conversion.
"""

import os
import json
import logging
import zipfile
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ExportConfig:
    """Configuration for export operations."""
    format: str = "json"
    include_metadata: bool = True
    include_plots: bool = True
    include_raw_data: bool = True
    compression: bool = False
    timestamp: bool = True
    author: str = ""
    title: str = ""
    description: str = ""
    output_dir: str = ""

class PhysicsDataExport:
    """Provides export and reporting functionality for physics data."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Initialize the Physics Data Export system.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory for export outputs
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path.home() / ".ai_research_lab" / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "data"
        self.archives_dir = self.output_dir / "archives"
        
        for directory in [self.reports_dir, self.data_dir, self.archives_dir]:
            directory.mkdir(exist_ok=True)
        
        # Export templates
        self.templates = self._load_export_templates()
        
        # Supported formats
        self.supported_formats = {
            'json': self._export_json,
            'csv': self._export_csv,
            'hdf5': self._export_hdf5,
            'xlsx': self._export_excel,
            'pdf': self._export_pdf,
            'latex': self._export_latex,
            'xml': self._export_xml,
            'archive': self._export_archive
        }
        
        logger.info(f"PhysicsDataExport initialized with output directory: {self.output_dir}")
    
    def _load_export_templates(self) -> Dict[str, str]:
        """Load export templates."""
        return {
            'latex_article': r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}

\title{{{title}}}
\author{{{author}}}
\date{{{date}}}

\begin{document}
\maketitle

\begin{abstract}
{abstract}
\end{abstract}

{content}

\end{document}
""",
            'html_report': """
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .figure {{ text-align: center; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p><strong>Author:</strong> {author}</p>
    <p><strong>Date:</strong> {date}</p>
    
    <h2>Abstract</h2>
    <p>{abstract}</p>
    
    {content}
</body>
</html>
"""
        }
    
    def export_physics_results(self, results: Dict[str, Any], format: str, **kwargs) -> Optional[str]:
        """
        Export physics results in specified format.
        
        Args:
            results: Physics results to export
            format: Export format
            **kwargs: Additional export parameters
        
        Returns:
            Path to exported file or None if failed
        """
        try:
            # Create export configuration
            export_config = ExportConfig(**{**self.config.get('export', {}), **kwargs})
            export_config.format = format
            
            # Validate format
            if format not in self.supported_formats:
                logger.error(f"Unsupported export format: {format}")
                return None
            
            # Add metadata
            if export_config.include_metadata:
                results = self._add_export_metadata(results, export_config)
            
            # Perform export
            exporter = self.supported_formats[format]
            output_path = exporter(results, export_config)
            
            if output_path:
                logger.info(f"Successfully exported results to: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export results: {str(e)}")
            return None
    
    def _add_export_metadata(self, results: Dict[str, Any], config: ExportConfig) -> Dict[str, Any]:
        """Add export metadata to results."""
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'export_format': config.format,
            'author': config.author,
            'title': config.title,
            'description': config.description,
            'framework_version': '1.0.0'
        }
        
        if 'metadata' not in results:
            results['metadata'] = {}
        
        results['metadata'].update(metadata)
        return results
    
    def _export_json(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as JSON."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_results_{timestamp}.json" if timestamp else "physics_results.json"
            output_path = self.data_dir / filename
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"JSON export failed: {str(e)}")
            return None
    
    def _export_csv(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as CSV."""
        if not PANDAS_AVAILABLE:
            logger.error("pandas required for CSV export")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            
            # Export different data types to separate CSV files
            output_paths = []
            
            for key, data in results.items():
                if key == 'metadata':
                    continue
                
                if isinstance(data, (list, dict)):
                    try:
                        # Try to convert to DataFrame
                        df = pd.DataFrame(data)
                        filename = f"{key}_{timestamp}.csv" if timestamp else f"{key}.csv"
                        output_path = self.data_dir / filename
                        df.to_csv(output_path, index=False)
                        output_paths.append(str(output_path))
                    except Exception:
                        # Skip if can't convert to DataFrame
                        continue
            
            # Return the first file path or None
            return output_paths[0] if output_paths else None
            
        except Exception as e:
            logger.error(f"CSV export failed: {str(e)}")
            return None
    
    def _export_hdf5(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as HDF5."""
        if not HDF5_AVAILABLE:
            logger.error("h5py required for HDF5 export")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_results_{timestamp}.h5" if timestamp else "physics_results.h5"
            output_path = self.data_dir / filename
            
            with h5py.File(output_path, 'w') as f:
                # Save metadata as attributes
                if 'metadata' in results:
                    for key, value in results['metadata'].items():
                        try:
                            f.attrs[key] = value
                        except (TypeError, ValueError):
                            f.attrs[key] = str(value)
                
                # Save data
                for key, data in results.items():
                    if key == 'metadata':
                        continue
                    
                    try:
                        if isinstance(data, (list, tuple)):
                            data = np.array(data) if NUMPY_AVAILABLE else data
                        
                        if isinstance(data, dict):
                            # Create group for dictionaries
                            group = f.create_group(key)
                            for subkey, subdata in data.items():
                                try:
                                    group.create_dataset(subkey, data=subdata)
                                except (TypeError, ValueError):
                                    group.create_dataset(subkey, data=str(subdata))
                        else:
                            f.create_dataset(key, data=data)
                    
                    except Exception as e:
                        logger.warning(f"Failed to save {key} to HDF5: {str(e)}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"HDF5 export failed: {str(e)}")
            return None
    
    def _export_excel(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as Excel file."""
        if not PANDAS_AVAILABLE:
            logger.error("pandas required for Excel export")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_results_{timestamp}.xlsx" if timestamp else "physics_results.xlsx"
            output_path = self.data_dir / filename
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Export metadata
                if 'metadata' in results:
                    metadata_df = pd.DataFrame(list(results['metadata'].items()), 
                                             columns=['Key', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                # Export data sheets
                for key, data in results.items():
                    if key == 'metadata':
                        continue
                    
                    try:
                        if isinstance(data, (list, dict)):
                            df = pd.DataFrame(data)
                            # Truncate sheet name to Excel limit
                            sheet_name = key[:31]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    except Exception as e:
                        logger.warning(f"Failed to export {key} to Excel: {str(e)}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Excel export failed: {str(e)}")
            return None
    
    def _export_pdf(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as PDF report."""
        if not REPORTLAB_AVAILABLE:
            logger.error("reportlab required for PDF export")
            return None
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_report_{timestamp}.pdf" if timestamp else "physics_report.pdf"
            output_path = self.reports_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = config.title or results.get('metadata', {}).get('title', 'Physics Results Report')
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=24,
                textColor=colors.darkblue,
                spaceAfter=30
            )
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Author and date
            author = config.author or results.get('metadata', {}).get('author', 'Unknown')
            date = datetime.now().strftime('%Y-%m-%d')
            story.append(Paragraph(f"<b>Author:</b> {author}", styles['Normal']))
            story.append(Paragraph(f"<b>Date:</b> {date}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Abstract/Description
            description = config.description or results.get('metadata', {}).get('description', '')
            if description:
                story.append(Paragraph("<b>Abstract</b>", styles['Heading2']))
                story.append(Paragraph(description, styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Results section
            story.append(Paragraph("<b>Results</b>", styles['Heading2']))
            
            # Add data tables
            for key, data in results.items():
                if key == 'metadata':
                    continue
                
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}</b>", styles['Heading3']))
                
                if isinstance(data, dict):
                    # Create table from dictionary
                    table_data = [['Parameter', 'Value']]
                    for k, v in data.items():
                        table_data.append([str(k), str(v)])
                    
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(table)
                    story.append(Spacer(1, 12))
                
                elif isinstance(data, list) and len(data) > 0:
                    # Try to create table from list
                    if isinstance(data[0], dict):
                        # List of dictionaries
                        if len(data) <= 50:  # Limit table size
                            headers = list(data[0].keys())
                            table_data = [headers]
                            for item in data:
                                row = [str(item.get(h, '')) for h in headers]
                                table_data.append(row)
                            
                            table = Table(table_data)
                            table.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, 0), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                ('GRID', (0, 0), (-1, -1), 1, colors.black)
                            ]))
                            story.append(table)
                        else:
                            story.append(Paragraph(f"Dataset contains {len(data)} entries (too large to display)", styles['Normal']))
                    else:
                        # Simple list
                        text = ', '.join(str(item) for item in data[:20])
                        if len(data) > 20:
                            text += f"... ({len(data)} total items)"
                        story.append(Paragraph(text, styles['Normal']))
                    
                    story.append(Spacer(1, 12))
                else:
                    # Simple value
                    story.append(Paragraph(str(data), styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(story)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"PDF export failed: {str(e)}")
            return None
    
    def _export_latex(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as LaTeX document."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_results_{timestamp}.tex" if timestamp else "physics_results.tex"
            output_path = self.reports_dir / filename
            
            # Prepare content
            content_sections = []
            
            for key, data in results.items():
                if key == 'metadata':
                    continue
                
                section_title = key.replace('_', ' ').title()
                content_sections.append(f"\\section{{{section_title}}}")
                
                if isinstance(data, dict):
                    # Create table
                    content_sections.append("\\begin{table}[H]")
                    content_sections.append("\\centering")
                    content_sections.append("\\begin{tabular}{|l|l|}")
                    content_sections.append("\\hline")
                    content_sections.append("Parameter & Value \\\\")
                    content_sections.append("\\hline")
                    
                    for k, v in data.items():
                        escaped_k = str(k).replace('_', '\\_').replace('%', '\\%')
                        escaped_v = str(v).replace('_', '\\_').replace('%', '\\%')
                        content_sections.append(f"{escaped_k} & {escaped_v} \\\\")
                    
                    content_sections.append("\\hline")
                    content_sections.append("\\end{tabular}")
                    content_sections.append(f"\\caption{{{section_title}}}")
                    content_sections.append("\\end{table}")
                
                elif isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        # Table from list of dicts (simplified)
                        content_sections.append("Data table with " + str(len(data)) + " entries.")
                    else:
                        # Simple list
                        items = ', '.join(str(item) for item in data[:10])
                        if len(data) > 10:
                            items += f" ... ({len(data)} total)"
                        content_sections.append(items)
                else:
                    content_sections.append(str(data))
                
                content_sections.append("")  # Empty line
            
            # Fill template
            template = self.templates['latex_article']
            content = template.format(
                title=config.title or 'Physics Results',
                author=config.author or 'Unknown',
                date=datetime.now().strftime('%Y-%m-%d'),
                abstract=config.description or 'Physics results report.',
                content='\n'.join(content_sections)
            )
            
            with open(output_path, 'w') as f:
                f.write(content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"LaTeX export failed: {str(e)}")
            return None
    
    def _export_xml(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as XML."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_results_{timestamp}.xml" if timestamp else "physics_results.xml"
            output_path = self.data_dir / filename
            
            def dict_to_xml(data, root_name='data'):
                """Convert dictionary to XML string."""
                xml_lines = [f"<{root_name}>"]
                
                for key, value in data.items():
                    safe_key = str(key).replace(' ', '_').replace('&', 'and')
                    
                    if isinstance(value, dict):
                        xml_lines.append(f"  <{safe_key}>")
                        for subkey, subvalue in value.items():
                            safe_subkey = str(subkey).replace(' ', '_').replace('&', 'and')
                            escaped_value = str(subvalue).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            xml_lines.append(f"    <{safe_subkey}>{escaped_value}</{safe_subkey}>")
                        xml_lines.append(f"  </{safe_key}>")
                    elif isinstance(value, list):
                        xml_lines.append(f"  <{safe_key}>")
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                xml_lines.append(f"    <item_{i}>")
                                for subkey, subvalue in item.items():
                                    safe_subkey = str(subkey).replace(' ', '_').replace('&', 'and')
                                    escaped_value = str(subvalue).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                    xml_lines.append(f"      <{safe_subkey}>{escaped_value}</{safe_subkey}>")
                                xml_lines.append(f"    </item_{i}>")
                            else:
                                escaped_value = str(item).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                xml_lines.append(f"    <item>{escaped_value}</item>")
                        xml_lines.append(f"  </{safe_key}>")
                    else:
                        escaped_value = str(value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        xml_lines.append(f"  <{safe_key}>{escaped_value}</{safe_key}>")
                
                xml_lines.append(f"</{root_name}>")
                return '\n'.join(xml_lines)
            
            # Create XML content
            xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
            xml_content += dict_to_xml(results, 'physics_results')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"XML export failed: {str(e)}")
            return None
    
    def _export_archive(self, results: Dict[str, Any], config: ExportConfig) -> Optional[str]:
        """Export results as compressed archive."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if config.timestamp else ''
            filename = f"physics_archive_{timestamp}.zip" if timestamp else "physics_archive.zip"
            output_path = self.archives_dir / filename
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Export to multiple formats in temp directory
                temp_config = config
                temp_config.timestamp = False
                
                exported_files = []
                
                # JSON export
                json_path = self._export_json(results, temp_config)
                if json_path:
                    shutil.copy2(json_path, temp_path / "results.json")
                    exported_files.append("results.json")
                
                # CSV export if pandas available
                if PANDAS_AVAILABLE:
                    csv_path = self._export_csv(results, temp_config)
                    if csv_path:
                        # Copy all CSV files
                        for csv_file in Path(csv_path).parent.glob("*.csv"):
                            if csv_file.name.endswith(".csv"):
                                shutil.copy2(csv_file, temp_path / csv_file.name)
                                exported_files.append(csv_file.name)
                
                # Create metadata file
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'exported_files': exported_files,
                    'archive_format': 'zip',
                    'compression': True,
                    'author': config.author,
                    'title': config.title,
                    'description': config.description
                }
                
                metadata_path = temp_path / "archive_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                exported_files.append("archive_metadata.json")
                
                # Create ZIP archive
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_name in exported_files:
                        file_path = temp_path / file_name
                        if file_path.exists():
                            zipf.write(file_path, file_name)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Archive export failed: {str(e)}")
            return None
    
    # Report generation methods
    def generate_manuscript_draft(self, research_data: Dict[str, Any], template: str = 'standard') -> Optional[str]:
        """Generate manuscript draft from research data."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"manuscript_draft_{timestamp}.tex"
            output_path = self.reports_dir / filename
            
            # Extract key information
            title = research_data.get('title', 'Research Results')
            abstract = research_data.get('abstract', 'This study presents research results.')
            introduction = research_data.get('introduction', 'Introduction section.')
            methods = research_data.get('methods', 'Methods section.')
            results = research_data.get('results', {})
            discussion = research_data.get('discussion', 'Discussion section.')
            conclusion = research_data.get('conclusion', 'Conclusion section.')
            
            # Create LaTeX manuscript
            manuscript_content = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{float}}
\\usepackage{{cite}}

\\title{{{title}}}
\\author{{Research Team}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{introduction}

\\section{{Methods}}
{methods}

\\section{{Results}}
{self._format_results_for_latex(results)}

\\section{{Discussion}}
{discussion}

\\section{{Conclusion}}
{conclusion}

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""
            
            with open(output_path, 'w') as f:
                f.write(manuscript_content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Manuscript generation failed: {str(e)}")
            return None
    
    def _format_results_for_latex(self, results: Dict[str, Any]) -> str:
        """Format results section for LaTeX."""
        latex_content = []
        
        for key, data in results.items():
            section_title = key.replace('_', ' ').title()
            latex_content.append(f"\\subsection{{{section_title}}}")
            
            if isinstance(data, dict):
                if 'value' in data and 'uncertainty' in data:
                    # Measurement with uncertainty
                    value = data['value']
                    uncertainty = data['uncertainty']
                    unit = data.get('unit', '')
                    latex_content.append(f"The measured value is ${value} \\pm {uncertainty}$ {unit}.")
                else:
                    # General dictionary
                    for k, v in data.items():
                        latex_content.append(f"{k}: {v}")
            
            elif isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict):
                    latex_content.append(f"The dataset contains {len(data)} measurements.")
                else:
                    latex_content.append(f"Values: {', '.join(map(str, data[:5]))}{'...' if len(data) > 5 else ''}")
            
            else:
                latex_content.append(str(data))
            
            latex_content.append("")  # Empty line
        
        return '\n'.join(latex_content)
    
    def generate_summary_report(self, datasets: List[Dict[str, Any]]) -> Optional[str]:
        """Generate summary report from multiple datasets."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"summary_report_{timestamp}.html"
            output_path = self.reports_dir / filename
            
            # Calculate summary statistics
            total_datasets = len(datasets)
            data_types = {}
            domains = {}
            
            for dataset in datasets:
                data_type = dataset.get('data_type', 'unknown')
                domain = dataset.get('domain', 'unknown')
                
                data_types[data_type] = data_types.get(data_type, 0) + 1
                domains[domain] = domains.get(domain, 0) + 1
            
            # Create HTML content
            html_content = self.templates['html_report'].format(
                title="Physics Data Summary Report",
                author="AI Research Lab",
                date=datetime.now().strftime('%Y-%m-%d'),
                abstract="This report summarizes the physics datasets and analyses.",
                content=f"""
                <h2>Dataset Summary</h2>
                <p>Total datasets: {total_datasets}</p>
                
                <h3>By Data Type</h3>
                <table>
                    <tr><th>Type</th><th>Count</th></tr>
                    {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in data_types.items())}
                </table>
                
                <h3>By Domain</h3>
                <table>
                    <tr><th>Domain</th><th>Count</th></tr>
                    {''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in domains.items())}
                </table>
                
                <h2>Dataset Details</h2>
                {''.join(self._format_dataset_for_html(i, dataset) for i, dataset in enumerate(datasets))}
                """
            )
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Summary report generation failed: {str(e)}")
            return None
    
    def _format_dataset_for_html(self, index: int, dataset: Dict[str, Any]) -> str:
        """Format dataset information for HTML."""
        name = dataset.get('name', f'Dataset {index + 1}')
        data_type = dataset.get('data_type', 'unknown')
        domain = dataset.get('domain', 'unknown')
        created_at = dataset.get('created_at', 'unknown')
        
        return f"""
        <h3>Dataset {index + 1}: {name}</h3>
        <p><strong>Type:</strong> {data_type}</p>
        <p><strong>Domain:</strong> {domain}</p>
        <p><strong>Created:</strong> {created_at}</p>
        """
    
    # Utility methods
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return list(self.supported_formats.keys())
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get statistics about exports."""
        try:
            # Count files in each directory
            data_files = len(list(self.data_dir.glob("*")))
            report_files = len(list(self.reports_dir.glob("*")))
            archive_files = len(list(self.archives_dir.glob("*")))
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file())
            
            return {
                'total_exports': data_files + report_files + archive_files,
                'data_exports': data_files,
                'reports': report_files,
                'archives': archive_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get export statistics: {str(e)}")
            return {}
    
    def cleanup_old_exports(self, days_to_keep: int = 30) -> int:
        """Clean up old export files."""
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_count = 0
            
            for file_path in self.output_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old export files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup exports: {str(e)}")
            return 0