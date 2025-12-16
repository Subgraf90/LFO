"""
Help Window for the LFO application.

Displays an HTML-based manual in a separate window.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QPushButton, QHBoxLayout
)
from pathlib import Path
import os


class HelpWindow(QDialog):
    """
    Dialog window for displaying the user manual.
    
    Uses QTextBrowser to display HTML content.
    """
    
    def __init__(self, parent=None):
        """
        Initializes the Help Window.
        
        Args:
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("LFO - User Manual")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        # Path to Help directory
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        # Help directory: LFO/Module_LFO/Modules_Help/
        self.help_dir = current_dir.parent / "Modules_Help"
        
        # Create Help directory if it doesn't exist
        if not self.help_dir.exists():
            self.help_dir.mkdir(parents=True, exist_ok=True)
            (self.help_dir / "images").mkdir(exist_ok=True)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # QTextBrowser für HTML-Inhalt
        self.text_browser = QTextBrowser(self)
        self.text_browser.setOpenExternalLinks(True)  # Externe Links öffnen
        
        # Stil für bessere Lesbarkeit
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
                font-size: 11pt;
                background-color: white;
                padding: 15px;
            }
        """)
        
        layout.addWidget(self.text_browser)
        
        # Button layout at the bottom
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        # Load manual
        self.load_manual()
    
    def load_manual(self):
        """
        Loads the default HTML manual file.
        
        Looks for manual_de.html in the Help directory.
        If not found, displays a placeholder message.
        """
        manual_path = self.help_dir / "manual_de.html"
        
        if manual_path.exists():
            try:
                # Load HTML file
                # QUrl.fromLocalFile() ensures relative image paths work
                url = QUrl.fromLocalFile(str(manual_path))
                self.text_browser.setSource(url)
                
                # Set base URL for relative paths (for images)
                base_url = QUrl.fromLocalFile(str(manual_path.parent) + os.sep)
                self.text_browser.document().setBaseUrl(base_url)
                
                # Update window title
                self.setWindowTitle("LFO - User Manual")
                
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Error Loading Manual",
                    f"The HTML file could not be loaded:\n{e}"
                )
        else:
            # Fallback: placeholder text
            placeholder_html = """
            <html>
            <head>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    h1 { color: #2c3e50; }
                    p { line-height: 1.6; }
                </style>
            </head>
            <body>
                <h1>LFO User Manual</h1>
                <p>Loading manual...</p>
                <p><em>Note: The file manual_de.html was not found.</em></p>
                <p>Please create the manual file in the directory:</p>
                <pre>{}</pre>
            </body>
            </html>
            """.format(manual_path)
            self.text_browser.setHtml(placeholder_html)
    
    def set_html_content(self, html_content: str):
        """
        Sets the HTML content directly.
        
        Args:
            html_content: HTML string with manual content
        """
        self.text_browser.setHtml(html_content)

