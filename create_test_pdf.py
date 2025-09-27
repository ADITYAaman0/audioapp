#!/usr/bin/env python3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_test_pdf():
    # Create a simple PDF for testing the PDF viewer
    c = canvas.Canvas("test_sample.pdf", pagesize=letter)
    width, height = letter
    
    # Page 1
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Advanced TTS Studio - PDF Test")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "This is a test PDF document to demonstrate the PDF viewer functionality.")
    c.drawString(50, height - 120, "The TTS app can now display PDF content before converting to speech.")
    
    c.drawString(50, height - 160, "Features of the PDF Viewer:")
    c.drawString(70, height - 180, "• Adjustable width and height")
    c.drawString(70, height - 200, "• Integrated with existing file upload")
    c.drawString(70, height - 220, "• Shows file information (name and size)")
    c.drawString(70, height - 240, "• Seamless integration with TTS workflow")
    
    c.drawString(50, height - 300, "Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    c.drawString(50, height - 320, "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
    c.drawString(50, height - 340, "Ut enim ad minim veniam, quis nostrud exercitation ullamco.")
    
    c.showPage()
    
    # Page 2
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Second Page")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "This is the second page of the test PDF.")
    c.drawString(50, height - 120, "You can navigate through pages using the PDF viewer controls.")
    
    c.drawString(50, height - 160, "The text from this PDF will be extracted and available")
    c.drawString(50, height - 180, "for text-to-speech conversion using the TTS functionality.")
    
    c.save()
    print("Test PDF created successfully: test_sample.pdf")

if __name__ == "__main__":
    create_test_pdf()