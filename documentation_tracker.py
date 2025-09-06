#!/usr/bin/env python3
"""
Documentation Tracker - DuckDB-powered documentation organization system

Leverages patterns from star-schema-llm-context to provide:
1. Structured metadata tracking for all documentation files
2. Relationship mapping between analyses
3. Timeline tracking with supersession detection
4. Organized retrieval for LLM consumption
"""

import os
import re
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict

@dataclass
class DocumentMetadata:
    """Structured metadata for documentation files"""
    file_id: str
    file_path: str
    file_name: str
    file_hash: str
    
    # Content analysis
    title: Optional[str]
    themes: List[str]
    analysis_type: str  # 'bug_report', 'implementation', 'comparison', 'analysis', etc.
    phase: Optional[str]  # 'phase_1', 'phase_2', etc.
    
    # Temporal tracking
    created_at: datetime
    modified_at: datetime
    analysis_date: Optional[datetime]  # Date of analysis content
    
    # Relationships
    references: List[str]  # Files referenced in this document
    supersedes: List[str]  # Files this analysis supersedes
    superseded_by: List[str]  # Files that supersede this analysis
    
    # Status
    status: str  # 'current', 'superseded', 'archived', 'draft'
    confidence_level: float  # 0.0-1.0 confidence in analysis
    
    # Metadata
    word_count: int
    conclusion_present: bool
    has_implementation_plan: bool
    has_timeline: bool

class DocumentationTracker:
    """SQLite-powered documentation organization and tracking system"""
    
    def __init__(self, base_path: str = ".", db_path: str = ".claude/documentation.db"):
        self.base_path = Path(base_path)
        self.documentation_path = self.base_path / "internal"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        self._local = threading.local()
        self._lock = threading.Lock()
        
        self._init_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        try:
            # SQLite uses autocommit mode by default, begin manually
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
    
    def _init_schema(self):
        """Initialize SQLite schema for documentation tracking"""
        conn = self._get_connection()
        
        # Documentation Files (Fact Table)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_files (
                file_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                
                -- Content analysis
                title TEXT,
                themes TEXT, -- JSON array as text
                analysis_type TEXT,
                phase TEXT,
                
                -- Temporal tracking
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                analysis_date TEXT,
                
                -- Status and quality
                status TEXT DEFAULT 'current',
                confidence_level REAL DEFAULT 1.00,
                
                -- Content metrics
                word_count INTEGER DEFAULT 0,
                conclusion_present BOOLEAN DEFAULT 0,
                has_implementation_plan BOOLEAN DEFAULT 0,
                has_timeline BOOLEAN DEFAULT 0
            )
        """)
        
        # Document Relationships (Edge Table)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_relationships (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                source_file_id TEXT NOT NULL,
                target_file_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL, -- 'references', 'supersedes', 'builds_on', 'contradicts'
                confidence REAL DEFAULT 1.00,
                detected_at TEXT DEFAULT (datetime('now')),
                
                FOREIGN KEY (source_file_id) REFERENCES doc_files(file_id),
                FOREIGN KEY (target_file_id) REFERENCES doc_files(file_id)
            )
        """)
        
        # Analysis Sessions (Dimension Table)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                session_date TEXT DEFAULT (datetime('now')),
                files_analyzed INTEGER DEFAULT 0,
                insights_generated INTEGER DEFAULT 0,
                conflicts_detected INTEGER DEFAULT 0,
                recommendations_made INTEGER DEFAULT 0
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_doc_files_path ON doc_files(file_path)",
            "CREATE INDEX IF NOT EXISTS idx_doc_files_type ON doc_files(analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_doc_files_status ON doc_files(status)",
            "CREATE INDEX IF NOT EXISTS idx_doc_files_modified ON doc_files(modified_at)",
            "CREATE INDEX IF NOT EXISTS idx_doc_relationships_source ON doc_relationships(source_file_id)",
            "CREATE INDEX IF NOT EXISTS idx_doc_relationships_target ON doc_relationships(target_file_id)",
            "CREATE INDEX IF NOT EXISTS idx_doc_relationships_type ON doc_relationships(relationship_type)"
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        conn.commit()
    
    def _generate_file_id(self, file_path: str) -> str:
        """Generate consistent file ID from path"""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            return "unknown"
    
    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract structured metadata from documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = ""
        
        # Basic file info
        file_id = self._generate_file_id(str(file_path))
        file_hash = self._calculate_file_hash(file_path)
        stat = file_path.stat()
        
        # Extract title (first # heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else file_path.stem.replace('_', ' ').title()
        
        # Detect themes and analysis type
        themes = self._detect_themes(content, file_path.name)
        analysis_type = self._classify_analysis_type(content, file_path.name, themes)
        
        # Extract phase information
        phase_match = re.search(r'(?:phase[_\s]*(\d+)|Phase\s+(\d+))', content, re.IGNORECASE)
        phase = f"phase_{phase_match.group(1) or phase_match.group(2)}" if phase_match else None
        
        # Detect analysis date from content
        date_patterns = [
            r'(?:Date|Created|Updated|Analysis):\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2})\s*Analysis',
            r'As of\s+(\d{4}-\d{2}-\d{2})'
        ]
        analysis_date = None
        for pattern in date_patterns:
            date_match = re.search(pattern, content, re.IGNORECASE)
            if date_match:
                try:
                    analysis_date = datetime.fromisoformat(date_match.group(1)).replace(tzinfo=timezone.utc)
                    break
                except:
                    pass
        
        # Extract file references
        references = self._extract_references(content)
        
        # Content analysis
        word_count = len(content.split())
        conclusion_present = bool(re.search(r'##?\s*Conclusion|##?\s*Summary|##?\s*Results', content, re.IGNORECASE))
        has_implementation_plan = bool(re.search(r'##?\s*Implementation|##?\s*Plan|##?\s*Roadmap', content, re.IGNORECASE))
        has_timeline = bool(re.search(r'##?\s*Timeline|Phase\s+\d+|Step\s+\d+', content, re.IGNORECASE))
        
        # Determine status and confidence
        status = self._determine_status(content, file_path.name)
        confidence_level = self._assess_confidence(content, conclusion_present, has_implementation_plan)
        
        return DocumentMetadata(
            file_id=file_id,
            file_path=str(file_path),
            file_name=file_path.name,
            file_hash=file_hash,
            title=title,
            themes=themes,
            analysis_type=analysis_type,
            phase=phase,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            analysis_date=analysis_date,
            references=references,
            supersedes=[],  # Will be populated by relationship analysis
            superseded_by=[],
            status=status,
            confidence_level=confidence_level,
            word_count=word_count,
            conclusion_present=conclusion_present,
            has_implementation_plan=has_implementation_plan,
            has_timeline=has_timeline
        )
    
    def _detect_themes(self, content: str, filename: str) -> List[str]:
        """Detect themes in documentation content"""
        themes = []
        
        theme_patterns = {
            "bugs": ["bug", "issue", "error", "fix", "problem", "broken"],
            "comparison": ["comparison", "vs", "versus", "compare", "three_way"],
            "implementation": ["implementation", "guide", "plan", "roadmap", "tutorial"],
            "analysis": ["analysis", "investigation", "deep", "study", "research"],
            "technical": ["technical", "parameters", "details", "specs", "api"],
            "multiframe": ["multiframe", "temporal", "frame", "sequence", "video"],
            "qwen": ["qwen", "qwenvl", "qwen2.5"],
            "comfyui": ["comfyui", "comfy", "ui"],
            "diffsynth": ["diffsynth", "diffusion", "synthesis"],
            "phase": ["phase", "milestone", "stage"],
            "architecture": ["architecture", "design", "structure", "pattern"],
            "performance": ["performance", "optimization", "speed", "memory"],
            "workflow": ["workflow", "pipeline", "process", "flow"]
        }
        
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in content_lower or keyword in filename_lower for keyword in keywords):
                themes.append(theme)
        
        return themes if themes else ["uncategorized"]
    
    def _classify_analysis_type(self, content: str, filename: str, themes: List[str]) -> str:
        """Classify the type of analysis document"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        # Priority-based classification
        if "bug" in themes or any(word in filename_lower for word in ["bug", "issue", "error", "fix"]):
            return "bug_report"
        elif "comparison" in themes or any(word in content_lower for word in ["vs", "versus", "compare"]):
            return "comparison"
        elif "implementation" in themes:
            return "implementation_guide"
        elif "analysis" in themes:
            return "deep_analysis"
        elif "technical" in themes:
            return "technical_documentation"
        elif "phase" in themes:
            return "phase_documentation"
        elif any(word in content_lower for word in ["conclusion", "summary", "results"]):
            return "summary_report"
        else:
            return "general_documentation"
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract references to other files from content"""
        references = []
        
        # Markdown link references
        md_links = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)
        for _, link in md_links:
            if not link.startswith('http'):
                references.append(link)
        
        # Direct file references
        file_refs = re.findall(r'`?([A-Z_][A-Z0-9_]*\.md)`?', content)
        references.extend(file_refs)
        
        # See also references
        see_also = re.findall(r'See\s+(?:also\s+)?(?:\[([^\]]+)\]|\*\*([^*]+)\*\*|`([^`]+)`)', content, re.IGNORECASE)
        for groups in see_also:
            for group in groups:
                if group and group.endswith('.md'):
                    references.append(group)
        
        return list(set(references))  # Remove duplicates
    
    def _determine_status(self, content: str, filename: str) -> str:
        """Determine current status of the document"""
        content_lower = content.lower()
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ["draft", "wip", "temp"]):
            return "draft"
        elif any(word in content_lower for word in ["superseded", "deprecated", "outdated"]):
            return "superseded"
        elif any(word in content_lower for word in ["archive", "legacy", "old"]):
            return "archived"
        else:
            return "current"
    
    def _assess_confidence(self, content: str, has_conclusion: bool, has_plan: bool) -> float:
        """Assess confidence level of the analysis"""
        base_confidence = 0.5
        
        # Boost for structured content
        if has_conclusion:
            base_confidence += 0.2
        if has_plan:
            base_confidence += 0.15
        
        # Boost for evidence of testing/validation
        if any(word in content.lower() for word in ["tested", "validated", "confirmed", "verified"]):
            base_confidence += 0.1
        
        # Penalty for uncertainty markers
        uncertainty_markers = len(re.findall(r'\b(?:maybe|perhaps|possibly|unclear|uncertain|TODO|FIXME)\b', content, re.IGNORECASE))
        base_confidence -= min(0.3, uncertainty_markers * 0.05)
        
        return max(0.1, min(1.0, base_confidence))
    
    def scan_documentation(self) -> Dict[str, Any]:
        """Scan all documentation files and update database"""
        if not self.documentation_path.exists():
            return {"error": "Internal documentation directory not found"}
        
        session_id = None
        files_processed = 0
        conflicts_detected = 0
        
        with self.transaction() as conn:
            # Create analysis session
            cursor = conn.execute("""
                INSERT INTO analysis_sessions (session_date) 
                VALUES (datetime('now'))
            """)
            session_id = cursor.lastrowid
            
            # Process all markdown files
            for md_file in self.documentation_path.glob("*.md"):
                try:
                    metadata = self._extract_metadata(md_file)
                    
                    # Insert or update file record
                    conn.execute("""
                        INSERT OR REPLACE INTO doc_files (
                            file_id, file_path, file_name, file_hash, title,
                            themes, analysis_type, phase, created_at, modified_at,
                            analysis_date, status, confidence_level, word_count,
                            conclusion_present, has_implementation_plan, has_timeline
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        metadata.file_id, metadata.file_path, metadata.file_name, metadata.file_hash,
                        metadata.title, json.dumps(metadata.themes), metadata.analysis_type, metadata.phase,
                        metadata.created_at.isoformat(), metadata.modified_at.isoformat(), 
                        metadata.analysis_date.isoformat() if metadata.analysis_date else None,
                        metadata.status, metadata.confidence_level, metadata.word_count,
                        metadata.conclusion_present, metadata.has_implementation_plan, metadata.has_timeline
                    ])
                    
                    files_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
                    continue
            
            # Update session stats
            conn.execute("""
                UPDATE analysis_sessions 
                SET files_analyzed = ?, conflicts_detected = ? 
                WHERE session_id = ?
            """, [files_processed, conflicts_detected, session_id])
        
        # Detect relationships and conflicts
        self._detect_relationships()
        
        return {
            "session_id": str(session_id),
            "files_processed": files_processed,
            "conflicts_detected": conflicts_detected,
            "timestamp": datetime.now().isoformat()
        }
    
    def _detect_relationships(self):
        """Detect and record relationships between documents"""
        with self.transaction() as conn:
            # Clear existing relationships
            conn.execute("DELETE FROM doc_relationships")
            
            # Get all files with their references
            files = conn.execute("""
                SELECT file_id, file_name, themes, analysis_type, created_at 
                FROM doc_files 
                WHERE status != 'archived'
            """).fetchall()
            
            for file_id, file_name, themes_json, analysis_type, created_at in files:
                themes = json.loads(themes_json) if themes_json else []
                # Find potential supersessions (similar analysis types, newer dates)
                similar_files = conn.execute("""
                    SELECT file_id, file_name, created_at
                    FROM doc_files 
                    WHERE file_id != ? 
                    AND analysis_type = ? 
                    AND status != 'archived'
                    ORDER BY created_at DESC
                """, [file_id, analysis_type]).fetchall()
                
                for other_id, other_name, other_created in similar_files:
                    if other_created > created_at:
                        # This file might be superseded by the newer one
                        confidence = 0.7  # Medium confidence for auto-detection
                        conn.execute("""
                            INSERT INTO doc_relationships 
                            (source_file_id, target_file_id, relationship_type, confidence)
                            VALUES (?, ?, 'supersedes', ?)
                        """, [other_id, file_id, confidence])
    
    def get_documentation_overview(self) -> Dict[str, Any]:
        """Get comprehensive overview of documentation status"""
        conn = self._get_connection()
        
        # Basic stats
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total_files,
                COUNT(CASE WHEN status = 'current' THEN 1 END) as current_files,
                COUNT(CASE WHEN status = 'superseded' THEN 1 END) as superseded_files,
                COUNT(CASE WHEN status = 'draft' THEN 1 END) as draft_files,
                AVG(confidence_level) as avg_confidence,
                SUM(word_count) as total_words
            FROM doc_files
        """).fetchone()
        
        # Theme distribution (simplified for SQLite)
        all_themes = {}
        theme_rows = conn.execute("""
            SELECT themes FROM doc_files WHERE status = 'current' AND themes IS NOT NULL
        """).fetchall()
        
        for (themes_json,) in theme_rows:
            if themes_json:
                themes_list = json.loads(themes_json)
                for theme in themes_list:
                    all_themes[theme] = all_themes.get(theme, 0) + 1
        
        themes = list(all_themes.items())
        
        # Analysis type distribution
        types = conn.execute("""
            SELECT analysis_type, COUNT(*) as count
            FROM doc_files
            WHERE status = 'current'
            GROUP BY analysis_type
            ORDER BY count DESC
        """).fetchall()
        
        # Recent activity
        recent = conn.execute("""
            SELECT file_name, modified_at, status
            FROM doc_files
            ORDER BY modified_at DESC
            LIMIT 10
        """).fetchall()
        
        # Supersession conflicts
        conflicts = conn.execute("""
            SELECT 
                s.file_name as superseded_file,
                t.file_name as superseding_file,
                r.confidence
            FROM doc_relationships r
            JOIN doc_files s ON r.target_file_id = s.file_id
            JOIN doc_files t ON r.source_file_id = t.file_id
            WHERE r.relationship_type = 'supersedes'
            AND r.confidence < 0.8
            ORDER BY r.confidence ASC
        """).fetchall()
        
        return {
            "stats": {
                "total_files": stats[0],
                "current_files": stats[1],
                "superseded_files": stats[2], 
                "draft_files": stats[3],
                "avg_confidence": float(stats[4]) if stats[4] else 0.0,
                "total_words": stats[5]
            },
            "themes": dict(themes),
            "analysis_types": dict(types),
            "recent_activity": [
                {"file": r[0], "modified": r[1], "status": r[2]} 
                for r in recent
            ],
            "potential_conflicts": [
                {
                    "superseded": r[0], 
                    "superseding": r[1], 
                    "confidence": float(r[2])
                } 
                for r in conflicts
            ]
        }
    
    def get_organized_documentation(self, theme: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get documentation organized by analysis lifecycle"""
        conn = self._get_connection()
        
        # Build query with theme filter
        if theme:
            files = conn.execute("""
                SELECT 
                    file_name, title, analysis_type, phase, themes,
                    status, confidence_level, modified_at, word_count,
                    conclusion_present, has_implementation_plan
                FROM doc_files
                WHERE status != 'archived'
                AND themes LIKE ?
                ORDER BY 
                    CASE analysis_type
                        WHEN 'deep_analysis' THEN 1
                        WHEN 'comparison' THEN 2 
                        WHEN 'implementation_guide' THEN 3
                        WHEN 'bug_report' THEN 4
                        WHEN 'technical_documentation' THEN 5
                        WHEN 'phase_documentation' THEN 6
                        ELSE 7
                    END,
                    modified_at DESC
            """, [f'%"{theme}"%']).fetchall()
        else:
            files = conn.execute("""
                SELECT 
                    file_name, title, analysis_type, phase, themes,
                    status, confidence_level, modified_at, word_count,
                    conclusion_present, has_implementation_plan
                FROM doc_files
                WHERE status != 'archived'
                ORDER BY 
                    CASE analysis_type
                        WHEN 'deep_analysis' THEN 1
                        WHEN 'comparison' THEN 2 
                        WHEN 'implementation_guide' THEN 3
                        WHEN 'bug_report' THEN 4
                        WHEN 'technical_documentation' THEN 5
                        WHEN 'phase_documentation' THEN 6
                        ELSE 7
                    END,
                    modified_at DESC
            """).fetchall()
        
        organized = {
            "Core Analysis": [],
            "Implementation Guides": [],
            "Comparisons & Studies": [],
            "Bug Reports & Issues": [],
            "Technical Documentation": [],
            "Phase Documentation": [],
            "Other": []
        }
        
        type_mapping = {
            "deep_analysis": "Core Analysis",
            "implementation_guide": "Implementation Guides", 
            "comparison": "Comparisons & Studies",
            "bug_report": "Bug Reports & Issues",
            "technical_documentation": "Technical Documentation",
            "phase_documentation": "Phase Documentation"
        }
        
        for file_data in files:
            category = type_mapping.get(file_data[2], "Other")
            themes = json.loads(file_data[4]) if file_data[4] else []
            organized[category].append({
                "file_name": file_data[0],
                "title": file_data[1],
                "analysis_type": file_data[2],
                "phase": file_data[3],
                "themes": themes,
                "status": file_data[5],
                "confidence": float(file_data[6]),
                "modified": file_data[7],  # Already a string from SQLite
                "word_count": file_data[8],
                "has_conclusion": bool(file_data[9]),
                "has_plan": bool(file_data[10])
            })
        
        return organized

if __name__ == "__main__":
    tracker = DocumentationTracker()
    
    print("Scanning documentation...")
    scan_result = tracker.scan_documentation()
    print(f"Processed {scan_result.get('files_processed', 0)} files")
    
    print("\nDocumentation Overview:")
    overview = tracker.get_documentation_overview()
    
    stats = overview['stats']
    print(f"  Total files: {stats['total_files']}")
    print(f"  Current: {stats['current_files']}, Superseded: {stats['superseded_files']}, Draft: {stats['draft_files']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Total words: {stats['total_words']:,}")
    
    print(f"\nTop themes: {', '.join(list(overview['themes'].keys())[:5])}")
    print(f"Potential conflicts: {len(overview['potential_conflicts'])}")
    
    # Save detailed report
    with open('.claude/documentation_report.json', 'w') as f:
        json.dump(overview, f, indent=2, default=str)
    
    print("\nDetailed report saved to .claude/documentation_report.json")