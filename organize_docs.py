#!/usr/bin/env python3
"""
Documentation Organization Utility

Simple CLI for organizing and querying documentation.
Built on SQLite-powered documentation tracking system.
"""

import sys
import json
from documentation_tracker import DocumentationTracker

def print_overview(tracker):
    """Print comprehensive documentation overview"""
    overview = tracker.get_documentation_overview()
    stats = overview['stats']
    
    print("Documentation Overview")
    print("=" * 50)
    print(f"Total files: {stats['total_files']}")
    print(f"Current: {stats['current_files']}, Superseded: {stats['superseded_files']}, Draft: {stats['draft_files']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"Total words: {stats['total_words']:,}")
    
    print(f"\nThemes ({len(overview['themes'])} total):")
    top_themes = sorted(overview['themes'].items(), key=lambda x: x[1], reverse=True)[:8]
    for theme, count in top_themes:
        print(f"  {theme}: {count} files")
    
    print(f"\nAnalysis Types:")
    for analysis_type, count in overview['analysis_types'].items():
        print(f"  {analysis_type}: {count} files")
    
    if overview['potential_conflicts']:
        print(f"\nPotential Conflicts ({len(overview['potential_conflicts'])} detected):")
        for conflict in overview['potential_conflicts'][:5]:  # Show first 5
            print(f"  {conflict['superseded']} -> {conflict['superseding']} (confidence: {conflict['confidence']})")
        if len(overview['potential_conflicts']) > 5:
            print(f"  ... and {len(overview['potential_conflicts']) - 5} more")

def print_organized_view(tracker, theme=None):
    """Print organized documentation by analysis lifecycle"""
    organized = tracker.get_organized_documentation(theme)
    
    if theme:
        print(f"Documentation organized by theme: {theme}")
    else:
        print("Documentation organized by analysis lifecycle")
    
    print("=" * 60)
    
    for category, files in organized.items():
        if files:
            print(f"\n{category} ({len(files)} files):")
            for file_info in files:
                confidence = file_info['confidence']
                status = file_info['status']
                word_count = file_info['word_count']
                
                status_marker = "[CURRENT]" if status == "current" else "[DRAFT]" if status == "draft" else "[SUPERSEDED]"
                confidence_level = "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.5 else "LOW"
                
                print(f"  {status_marker} {file_info['file_name']}")
                print(f"     {file_info['title']}")
                print(f"     Confidence: {confidence_level} ({confidence:.2f}) | Words: {word_count} | {', '.join(file_info['themes'][:3])}")

def search_by_theme(tracker, theme):
    """Search documentation by specific theme"""
    print(f"Searching for theme: {theme}")
    organized = tracker.get_organized_documentation(theme)
    
    total_files = sum(len(files) for files in organized.values())
    if total_files == 0:
        print(f"No files found for theme '{theme}'")
        return
    
    print(f"Found {total_files} files matching theme '{theme}':")
    print_organized_view(tracker, theme)

def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python organize_docs.py overview          - Show documentation overview")
        print("  python organize_docs.py organized         - Show organized view by lifecycle")
        print("  python organize_docs.py theme <name>      - Search by theme")
        print("  python organize_docs.py scan              - Scan and update documentation database")
        print("  python organize_docs.py themes            - List all available themes")
        return
    
    command = sys.argv[1].lower()
    tracker = DocumentationTracker()
    
    try:
        if command == "overview":
            print_overview(tracker)
        
        elif command == "organized":
            print_organized_view(tracker)
        
        elif command == "theme":
            if len(sys.argv) < 3:
                print("Usage: python organize_docs.py theme <theme_name>")
                return
            theme = sys.argv[2]
            search_by_theme(tracker, theme)
        
        elif command == "scan":
            print("Scanning documentation...")
            result = tracker.scan_documentation()
            print(f"Scanned {result['files_processed']} files")
            print("Updated documentation database")
        
        elif command == "themes":
            overview = tracker.get_documentation_overview()
            print("Available themes:")
            themes = sorted(overview['themes'].items(), key=lambda x: x[1], reverse=True)
            for theme, count in themes:
                print(f"  {theme} ({count} files)")
        
        elif command == "export":
            overview = tracker.get_documentation_overview()
            organized = tracker.get_organized_documentation()
            
            export_data = {
                "overview": overview,
                "organized": organized,
                "timestamp": tracker.scan_documentation()["timestamp"]
            }
            
            with open('.claude/documentation_export.json', 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print("Exported documentation data to .claude/documentation_export.json")
        
        else:
            print(f"Unknown command: {command}")
            print("Run without arguments to see usage help")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()