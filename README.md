# AI Invoice Generator

A smart web app that turns images into professional PDF invoices. Just upload a screenshot, handwritten note, or photo - and get an instant invoice.

## What it does

Upload any image with text (like a WhatsApp chat about an iPhone order, a handwritten note, or a phone screenshot), and the app will:

1. Read the text from your image using OCR
2. Find the customer name, address, and order details
3. Generate a clean PDF invoice ready to download

Perfect for small businesses, freelancers, or anyone who needs quick invoices from casual orders.

Working locally with using both methods regex and ollama slm model 

Regex works fine for quik simple examples 

Using SLM Moddel for complex extraction tasks 


## How to run locally

### Quick start with Docker

```bash
docker-compose up -d --build
```

Then open http://localhost:5000 in your browser.

### Or run locally with Python

```bash
# Install Tesseract first
sudo apt install tesseract-ocr tesseract-ocr-fra

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

## Features

- **Modern dark UI** - Clean chatbot-style interface
- **Dual extraction modes** - Choose between fast regex or AI (if Ollama is running)
- **Smart parsing** - Automatically finds names, addresses, quantities, and totals
- **Professional PDFs** - Well-formatted invoices with all details
- **Works with messy data** - Handles conversations, handwriting, notes

## Tech stack

- Flask (Python web framework)
- Tesseract OCR (text extraction from images)
- ReportLab (PDF generation)
- Vanilla JavaScript (no frameworks, fast and simple)
- Modern CSS (dark theme with smooth animations)

## Deploy anywhere

Works on any platform that supports Docker:
- Railway
- Render  

I have choose Render. You will find a quick demo here :
https://test-ai-front.onrender.com/

On the Render server : 

Only Regex works, the ressources are not capable of using ollama. Code in main is adjusted for that 

you will find the code using ollama on the develop branch 

## Made for

Built as a technical test to demonstrate:
- Clean code structure
- Modern UI/UX design
- Full-stack developmentp skills


---

That's it! Simple tool, clean code, gets the job done


Editor: Hedi Amine Chaabani 


Date: 04/11/2025

