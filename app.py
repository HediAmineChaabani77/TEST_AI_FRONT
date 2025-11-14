"""
Test AI Frontend for OCR Invoice Generator - Gemini Only Version
Flask Backend for OCR Invoice Generator
HTML/CSS frontend with a simple interface to upload an image and get the invoice data
Gemini ONLY version - uses Google Gemini 2.5 Flash for invoice data extraction (no regex, no ollama)
"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
from io import BytesIO

# OCR and Image Processing
try:
    from PIL import Image
    import pytesseract
    OCR_ENGINE = 'tesseract'
except ImportError:
    try:
        import easyocr
        OCR_ENGINE = 'easyocr'
        reader = easyocr.Reader(['fr', 'en'])
    except ImportError:
        OCR_ENGINE = None

# LLM for intelligent interpretation - Google Gemini ONLY
try:
    from google import genai
    LLM_AVAILABLE = True
    GEMINI_API_KEY = "AIzaSyCkpUCP0G_3XGHmAn_l7005GdaTpzadDVY"
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✓ Google Gemini available")
except ImportError:
    LLM_AVAILABLE = False
    gemini_client = None
    print("✗ ERROR: Google Gemini not available! This version REQUIRES Gemini.")
    print("  Install: pip install google-genai")

# PDF Generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfgen import canvas
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['INVOICE_FOLDER'] = 'invoices'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['INVOICE_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        if OCR_ENGINE == 'tesseract':
            text = pytesseract.image_to_string(img, lang='fra+eng')
        elif OCR_ENGINE == 'easyocr':
            results = reader.readtext(image_path)
            text = '\n'.join([result[1] for result in results])
        else:
            return None, "OCR engine not available"
        return text.strip(), None
    except Exception as e:
        return None, f"Error during OCR: {str(e)}"


def interpret_with_gemini(extracted_text):
    """
    Use Google Gemini 2.5 Flash to intelligently extract invoice data with structured output
    This is the ONLY method used in this version - no regex, no ollama fallback
    """
    if not LLM_AVAILABLE or gemini_client is None:
        return None, "Gemini not available - this version requires Gemini to function"
    
    try:
        # Define JSON schema for structured output
        invoice_schema = {
            "type": "object",
            "properties": {
                "client_name": {
                    "type": "string",
                    "description": "Client's full name (first name + last name) found in the text"
                },
                "client_address": {
                    "type": "string",
                    "description": "Client's address (street + postal code + city) if found in the text"
                },
                "items": {
                    "type": "array",
                    "description": "List of products/items found in the invoice",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Product name with color information if applicable (e.g., 'iPhone 14 (noir x1, blanc x1)')"
                            },
                            "quantity": {
                                "type": "integer",
                                "description": "Total quantity of the product"
                            },
                            "unit_price": {
                                "type": "number",
                                "description": "Unit price if found in text, otherwise null"
                            },
                            "total_price": {
                                "type": "number",
                                "description": "Total price for this item if found in text, otherwise null"
                            }
                        },
                        "required": ["description", "quantity"]
                    }
                },
                "invoice_total": {
                    "type": "number",
                    "description": "Total invoice amount if found in text, otherwise null"
                }
            },
            "required": ["client_name", "client_address", "items"]
        }
        
        prompt = f"""
Analysez le texte ci-dessous et extrayez les informations de facture en suivant strictement les règles.

TEXTE :
{extracted_text}

RÈGLES GÉNÉRALES :

1. Informations client
   - "client_name" = prénom + nom trouvés dans le texte.
   - "client_address" = rue + code postal + ville si trouvés.
   - Si une information n'est pas trouvée → laisser vide (mais ne rien inventer).

2. Produits (items)
   - Chaque entrée dans "items" = un produit trouvé dans le texte.
   - Ne jamais créer ou inventer de produits.
   - Ne jamais inventer de quantité : compter uniquement ce qui est dans le texte.

3. Extraction des prix
   - Extraire le prix unitaire si présent.
   - Extraire le prix total si présent.
   - Si le texte mentionne un total général pour la facture → le mettre dans "invoice_total".
   - Ne jamais utiliser de prix "habituels" ou supposés. Seulement ceux écrits dans le texte.
   - Ne pas déduire ou imaginer un prix s'il n'est pas explicitement indiqué.

4. RÈGLES COULEURS – TRÈS IMPORTANT POUR ÉVITER LES ERREURS :
   A. Si une quantité globale Q est associée au produit (ex: "2 iPhone 14"),
      ALORS la quantité totale = Q, même si plusieurs couleurs sont listées.
      → Ne jamais faire total = Q + (couleurs).
   
   B. Répartition par couleur :
      - Si des chiffres sont indiqués par couleur, les additionner SANS dépasser la quantité totale Q.
      - Si Q est indiqué mais qu'une couleur a un chiffre en plus (ex: "2 iPhone 14, couleurs: noir, 1 blanc"),
        répartition = noir x1, blanc x1 → total = 2. Pas plus.
      - Si Q n'est PAS indiqué :
        - S'il y a des quantités dans les couleurs → total = somme.
        - S'il n'y a pas de quantités dans les couleurs → total = nombre de couleurs.

   C. Cas à traiter correctement :
      - "2 iPhone 14, couleurs: (noir, 1 blanc" → total = 2 → noir x1, blanc x1.
      - "3 iPhone 14 (noir, rouge, bleu)" → total = 3.
      - "iPhone 14 (noir x2, blanc x1)" → total = 3.
      - "5 iPhone 14 (noir x2, blanc x4)" → total = 5 (tronquer à 5).

   D. Dans le JSON, pour la clarté :
      - Utiliser une seule ligne par produit.
      - Ajouter la répartition couleur dans "description", ex:
        "description": "iPhone 14 (noir x1, blanc x1)"

Retournez UNIQUEMENT les données extraites au format JSON structuré selon le schéma fourni.
Ne rien inventer. Utilisez uniquement les informations présentes explicitement dans le texte.
"""
        
        # Call Gemini with structured output
        # Try different API formats based on what works
        response = None
        last_error = None
        
        # Method 1: Try with config parameter (structured output)
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash-preview-09-2025",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": invoice_schema,
                    "temperature": 0.1,
                }
            )
        except Exception as e1:
            last_error = str(e1)
            print(f"Method 1 (config) failed: {e1}")
            
            # Method 2: Try with parameters dict
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash-preview-09-2025",
                    contents=prompt,
                    parameters={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    }
                )
            except Exception as e2:
                last_error = str(e2)
                print(f"Method 2 (parameters) failed: {e2}")
                
                # Method 3: Try simple call (like user's example)
                try:
                    response = gemini_client.models.generate_content(
                        model="gemini-2.5-flash-preview-09-2025",
                        contents=prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching this schema: " + json.dumps(invoice_schema) + "\nNo additional text, just JSON.",
                    )
                except Exception as e3:
                    last_error = str(e3)
                    print(f"Method 3 (simple) failed: {e3}")
                    
                    # Method 4: Try with shorter model name
                    try:
                        response = gemini_client.models.generate_content(
                            model="gemini-2.5-flash-preview",
                            contents=prompt + "\n\nIMPORTANT: Return ONLY valid JSON matching this schema: " + json.dumps(invoice_schema) + "\nNo additional text, just JSON.",
                        )
                    except Exception as e4:
                        last_error = str(e4)
                        print(f"Method 4 (short model name) failed: {e4}")
                        return None, f"All Gemini API call methods failed. Last error: {last_error}"
        
        if response is None:
            return None, f"Failed to get response from Gemini. Last error: {last_error}"
        
        # Parse the structured JSON response
        response_text = None
        
        # Debug: print response type and attributes
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Try different ways to extract text from response
        if hasattr(response, 'text'):
            response_text = response.text
            print(f"Got text from response.text: {response_text[:200] if response_text else 'None'}...")
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            # Alternative response format
            content = response.candidates[0].content
            if hasattr(content, 'parts') and len(content.parts) > 0:
                response_text = content.parts[0].text
                print(f"Got text from candidates[0].content.parts[0].text: {response_text[:200] if response_text else 'None'}...")
            elif hasattr(content, 'text'):
                response_text = content.text
                print(f"Got text from candidates[0].content.text: {response_text[:200] if response_text else 'None'}...")
        elif hasattr(response, '__str__'):
            # Last resort: try to convert to string
            response_text = str(response)
            print(f"Got text from str(response): {response_text[:200] if response_text else 'None'}...")
        
        if not response_text:
            return None, f"No response text from Gemini. Response object: {type(response)}, attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}"
        
        # Extract JSON from response (handle cases where there might be extra text)
        response_text = response_text.strip()
        print(f"Full response text length: {len(response_text)}")
        
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            try:
                llm_data = json.loads(json_str)
                print("Successfully parsed JSON from response")
            except json.JSONDecodeError as je:
                return None, f"Failed to parse JSON from response: {str(je)}\nJSON string: {json_str[:500]}"
        else:
            # Try parsing the whole response as JSON
            try:
                llm_data = json.loads(response_text)
                print("Successfully parsed entire response as JSON")
            except json.JSONDecodeError as je:
                return None, f"Failed to parse response as JSON: {str(je)}\nResponse text: {response_text[:500]}"
        
        # Validate and fix items
        items = llm_data.get('items', [])
        if not items:
            return None, "No items found in Gemini response"
        
        # Ensure correct pricing and totals
        total = 0.0
        for item in items:
            # Ensure quantity is an integer
            if 'quantity' in item:
                item['quantity'] = int(item['quantity'])
            else:
                item['quantity'] = 1
            
            # Ensure unit_price is set (250 for iPhone, or use extracted value)
            description = item.get('description', '').lower()
            if 'iphone' in description:
                # Use extracted unit_price if available, otherwise default to 250
                if 'unit_price' not in item or item.get('unit_price') is None:
                    item['unit_price'] = 250.0
            elif 'unit_price' not in item or item.get('unit_price') is None:
                # For non-iPhone items, try to use total_price / quantity if available
                if 'total_price' in item and item.get('total_price') is not None:
                    item['unit_price'] = item['total_price'] / item['quantity'] if item['quantity'] > 0 else 0.0
                else:
                    item['unit_price'] = 250.0  # Default fallback
            
            # Calculate item total if not provided
            if 'total_price' in item and item.get('total_price') is not None:
                item['total'] = item['total_price']
            else:
                item['total'] = item['quantity'] * item['unit_price']
            
            total += item['total']
        
        # Use invoice_total if provided, otherwise use calculated total
        if 'invoice_total' in llm_data and llm_data.get('invoice_total') is not None:
            total = float(llm_data['invoice_total'])
        
        # Build invoice data
        invoice_data = {
            'client_name': llm_data.get('client_name', 'Client'),
            'client_address': llm_data.get('client_address', ''),
            'date': datetime.now().strftime('%d/%m/%Y'),
            'invoice_number': f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'items': items,
            'total': total,
            'description': ''
        }
        
        return invoice_data, None
            
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Gemini error: {str(e)}"


def generate_pdf_invoice(invoice_data, output_path):
    """Generate a professional PDF invoice with structured layout"""
    
    # Custom document class to add header/footer
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            canvas.Canvas.__init__(self, *args, **kwargs)
            self._saved_page_states = []
        
        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()
        
        def save(self):
            """add page info to each page (page x of y)"""
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_number(num_pages)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)
        
        def draw_page_number(self, page_count):
            # Draw header elements
            self.saveState()
            
            # Draw FACTURE title on the left
            self.setFont("Helvetica-Bold", 28)
            self.setFillColor(colors.HexColor('#2c3e50'))
            self.drawString(2*cm, A4[1] - 2.5*cm, "FACTURE")
            
            # Draw invoice number box (light gray background with dark border) - under FACTURE on left, wider to fit longer numbers
            inv_num_x = 2*cm
            inv_num_y = A4[1] - 3.5*cm
            # Light gray fill - box is wider (left to right) to accommodate longer invoice numbers
            self.setFillColor(colors.HexColor('#f5f5f5'))
            self.setStrokeColor(colors.HexColor('#2c3e50'))
            self.setLineWidth(1)
            self.roundRect(inv_num_x, inv_num_y, 8.5*cm, 0.8*cm, 0.3*cm, fill=1, stroke=1)  # Wider: 3.5->8.5cm, height stays 0.8cm
            
            # Invoice number text
            self.setFillColor(colors.HexColor('#2c3e50'))
            self.setFont("Helvetica", 10)
            inv_num = invoice_data.get('invoice_number', 'N/A')
            # Keep the format as is
            if not inv_num.startswith('INV-'):
                inv_num_display = f"INV-{inv_num}"
            else:
                inv_num_display = inv_num
            self.drawString(inv_num_x + 0.2*cm, inv_num_y + 0.25*cm, f"Facture n°{inv_num_display}")
            
            # Date box (solid dark box) - below invoice number, still on left
            date_x = 2*cm
            date_y = A4[1] - 4.5*cm
            self.setFillColor(colors.HexColor('#2c3e50'))
            self.setStrokeColor(colors.HexColor('#2c3e50'))
            self.roundRect(date_x, date_y, 2.5*cm, 0.8*cm, 0.3*cm, fill=1, stroke=1)
            self.setFillColor(colors.white)
            self.setFont("Helvetica", 10)
            date_str = invoice_data.get('date', datetime.now().strftime('%d/%m/%y'))
            self.drawString(date_x + 0.2*cm, date_y + 0.25*cm, date_str)
            
            # Draw logo/image placeholder on far top right (circle only) - original size
            logo_x = 16.5*cm
            logo_y = A4[1] - 2.2*cm
            logo_size = 1.5*cm  # Original size
            # Draw a circle outline (placeholder for logo)
            self.setStrokeColor(colors.HexColor('#2c3e50'))
            self.setFillColor(colors.white)
            self.setLineWidth(2)
            # Simple circle as placeholder - can be replaced with actual image later
            self.circle(logo_x + logo_size/2, logo_y + logo_size/2, logo_size/2, fill=0, stroke=1)
            
            self.restoreState()
    
    doc = SimpleDocTemplate(output_path, pagesize=A4, 
                          rightMargin=2*cm, leftMargin=2*cm,
                          topMargin=4*cm, bottomMargin=3*cm)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    sender_style = ParagraphStyle(
        'SenderStyle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold',
        spaceAfter=5
    )
    
    recipient_header_style = ParagraphStyle(
        'RecipientHeader',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold',
        spaceAfter=8
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=3
    )
    
    # Header section (will be drawn by canvas)
    story.append(Spacer(1, 0.5*cm))
    
    # Sender and Recipient information in two columns
    # Sender name should be bold and match template style
    sender_name = invoice_data.get('sender_name', 'HEDI')
    sender_text = f"""<b>{sender_name}</b><br/>
{invoice_data.get('sender_phone', 'Tel : 123-456-7890')}<br/>
{invoice_data.get('sender_email', 'hello@reallygreatsite.com')}<br/>
{invoice_data.get('sender_website', 'reallygreatsite.com')}<br/>
{invoice_data.get('sender_address', '123 Anywhere St., Any City')}"""
    sender_info = Paragraph(sender_text, normal_style)
    
    recipient_name = invoice_data.get('client_name', 'Client')
    recipient_address = invoice_data.get('client_address', '')
    recipient_phone = invoice_data.get('client_phone', '')
    
    recipient_text = f"""<b>À L'ATTENTION DE</b><br/>
<b>{recipient_name}</b>"""
    if recipient_address:
        recipient_text += f"<br/>{recipient_address}"
    if recipient_phone:
        recipient_text += f"<br/>{recipient_phone}"
    
    recipient_info = Paragraph(recipient_text, normal_style)
    
    # Create two-column table for sender/recipient
    info_table_data = [
        [sender_info, recipient_info]
    ]
    info_table = Table(info_table_data, colWidths=[9*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (0, -1), 10),
        ('RIGHTPADDING', (0, 0), (0, -1), 10),
        ('LEFTPADDING', (1, 0), (1, -1), 10),
        ('RIGHTPADDING', (1, 0), (1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 1*cm))
    
    # Items table with proper columns: DESCRIPTION, PRIX, QUANTITÉ, TOTAL
    items_data = [['DESCRIPTION', 'PRIX', 'QUANTITÉ', 'TOTAL']]
    
    # Calculate subtotal
    subtotal = 0.0
    for item in invoice_data['items']:
        item_total = item.get('total', item.get('quantity', 1) * item.get('unit_price', 0))
        subtotal += item_total
        items_data.append([
            item['description'],
            f"{item.get('unit_price', 0):.2f} €",
            f"{item.get('quantity', 1):02d}",
            f"{item_total:.2f} €"
        ])
    
    # Calculate VAT (set to 0 for now)
    vat_rate = 0.0
    vat_amount = 0.0
    total_with_vat = subtotal
    
    # Use total from invoice_data if provided, otherwise use calculated total
    final_total = invoice_data.get('total', total_with_vat)
    
    # Create items table
    items_table = Table(items_data, colWidths=[8*cm, 3*cm, 2.5*cm, 3.5*cm])
    items_table.setStyle(TableStyle([
        # Header row - dark gray background
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        
        # Data rows
        ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('ALIGN', (2, 1), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e0e0e0')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(items_table)
    story.append(Spacer(1, 0.5*cm))
    
    # Summary section (Subtotal, VAT, Total) - right aligned
    summary_data = [
        ['Sous total', f"{subtotal:,.2f} €".replace(',', ' ')],
        ['TVA (20%)', f"{vat_amount:,.2f} €".replace(',', ' ')],
        ['TOTAL', f"{final_total:,.2f} €".replace(',', ' ')],
    ]
    
    summary_table = Table(summary_data, colWidths=[4*cm, 4*cm])
    summary_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONT', (0, 0), (0, -1), 'Helvetica', 10),
        ('FONT', (1, 0), (1, -1), 'Helvetica-Bold', 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        # Total row - dark gray background
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
        ('FONTSIZE', (0, -1), (-1, -1), 11),
        ('TOPPADDING', (0, -1), (-1, -1), 10),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
    ]))
    
    # Right-align the summary table
    summary_wrapper = Table([[summary_table]], colWidths=[18*cm])
    summary_wrapper.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
    ]))
    story.append(summary_wrapper)
    story.append(Spacer(1, 1.5*cm))
    
    # Footer section - Payment information
    footer_data = [
        [
            Paragraph(f"<b>Paiement à l'ordre de</b><br/>{sender_name}<br/>N° de compte 0123 4567 8901 2345", normal_style),
            Paragraph(f"<b>Conditions de paiement</b><br/>Paiement sous 30 jours", normal_style)
        ]
    ]
    footer_table = Table(footer_data, colWidths=[9*cm, 9*cm])
    footer_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(footer_table)
    story.append(Spacer(1, 1*cm))
    
    # Thank you message - centered
    thank_you = Paragraph("<b>MERCI DE VOTRE CONFIANCE</b>", ParagraphStyle(
        'ThankYou',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=0
    ))
    story.append(thank_you)
    
    # Build PDF with custom canvas
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/check-gemini', methods=['GET'])
def check_gemini():
    """Check if Gemini is available"""
    return jsonify({
        'available': LLM_AVAILABLE,
        'model': 'gemini-2.5-flash-preview-09-2025' if LLM_AVAILABLE else None
    })


@app.route('/api/process-image', methods=['POST'])
def process_image():
    """
    Main endpoint: receives image, performs OCR, interprets data with Gemini, generates PDF
    Gemini ONLY version - no fallback methods
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Type de fichier non autorisé'}), 400
        
        # Check if Gemini is available - REQUIRED in this version
        if not LLM_AVAILABLE:
            return jsonify({'error': 'Google Gemini n\'est pas disponible. Cette version nécessite Gemini pour fonctionner.'}), 503
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        extracted_text, error = extract_text_from_image(filepath)
        if error:
            return jsonify({'error': error}), 500
        
        if not extracted_text:
            return jsonify({'error': 'Aucun texte détecté dans l\'image'}), 400
        
        # Always use Gemini for interpretation (ONLY method in this version)
        invoice_data, error = interpret_with_gemini(extracted_text)
        if error:
            return jsonify({'error': f'Erreur lors de l\'interprétation avec Gemini: {error}'}), 500
        
        method_used = 'gemini'
        
        pdf_filename = f"invoice_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['INVOICE_FOLDER'], pdf_filename)
        generate_pdf_invoice(invoice_data, pdf_path)
        
        with open(pdf_path, 'rb') as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'extracted_text': extracted_text,
            'invoice_data': invoice_data,
            'pdf_filename': pdf_filename,
            'pdf_base64': pdf_base64,
            'method_used': method_used
        })
    
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500


@app.route('/api/download/<filename>')
def download_invoice(filename):
    try:
        filepath = os.path.join(app.config['INVOICE_FOLDER'], secure_filename(filename))
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("🧾 OCR Invoice Generator - Gemini ONLY Version")
    print("=" * 60)
    print(f"✓ OCR Engine: {OCR_ENGINE if OCR_ENGINE else 'Not available'}")
    if LLM_AVAILABLE:
        print("✓ LLM: Google Gemini 2.5 Flash Preview (REQUIRED)")
    else:
        print("✗ ERROR: Google Gemini not available!")
        print("  This version REQUIRES Gemini to function.")
        print("  Install: pip install google-genai")
    print("=" * 60)
    print("🌐 Server running at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
