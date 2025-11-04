"""
Flask Backend for OCR Invoice Generator
Production version - Regex only (Ollama as optional fallback)
"""
import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
from io import BytesIO
import re

# OCR and Image Processing
try:
    from PIL import Image
    import pytesseract
    OCR_ENGINE = 'tesseract'
except ImportError:
    OCR_ENGINE = None

# LLM optional
try:
    import ollama
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# PDF Generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT

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
        else:
            return None, "OCR engine not available"
        return text.strip(), None
    except Exception as e:
        return None, f"Error during OCR: {str(e)}"


def interpret_with_ollama(extracted_text):
    if not LLM_AVAILABLE:
        return None, "Ollama not available"
    
    try:
        prompt = f"""
Analysez le texte ci-dessous et créez une facture AU FORMAT JSON UNIQUEMENT.
Ne pas écrire de texte en dehors du JSON. Ne rien inventer. N'utiliser que les informations présentes explicitement dans le texte.

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
      - "2 iPhone 14, couleurs:  (noir, 1 blanc" → total = 2 → noir x1, blanc x1.
      - "3 iPhone 14 (noir, rouge, bleu)" → total = 3.
      - "iPhone 14 (noir x2, blanc x1)" → total = 3.
      - "5 iPhone 14 (noir x2, blanc x4)" → total = 5 (tronquer à 5).
   
   D. Dans le JSON, pour la clarté :
      - Utiliser une seule ligne par produit.
      - Ajouter la répartition couleur dans "description", ex:
        "description": "iPhone 14 (noir x1, blanc x1)"

5. FORMAT JSON EXACT :
{{
    "client_name": "",
    "client_address": "",
    "items": [
        {{
            "description": "nom du produit (couleur)",
            "quantity": nombre_total,
            "unit_price": prix_unitaire_ou_null,
            "total_price": prix_total_ou_null
        }}
    ],
    "invoice_total": prix_total_facture_ou_null
}}

JSON uniquement.
"""

        response = ollama.generate(
            model='llama3.2:1b',
            prompt=prompt,
            stream=False,
            options={'temperature': 0.1, 'num_predict': 400}
        )
        
        response_text = response['response'].strip()
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            llm_data = json.loads(json_str)
            
            items = llm_data.get('items', [])
            if not items:
                return None, "No items found"
            
            total = 0.0
            for item in items:
                item['quantity'] = int(item.get('quantity', 1))
                if 'iphone' in item.get('description', '').lower():
                    item['unit_price'] = 250.0
                elif 'unit_price' not in item:
                    item['unit_price'] = 250.0
                item['total'] = item['quantity'] * item['unit_price']
                total += item['total']
            
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
        else:
            return None, "Invalid JSON"
    except Exception as e:
        return None, f"LLM error: {str(e)}"


def interpret_invoice_data(extracted_text):
    """Regex extraction - primary method"""
    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
    full_text = ' '.join(lines)
    
    invoice_data = {
        'client_name': '',
        'client_address': '',
        'date': datetime.now().strftime('%d/%m/%Y'),
        'invoice_number': f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'items': [],
        'total': 0.0,
        'description': ''
    }
    
    # Extract address
    address_pattern = r'(\d+\s+(?:rue|avenue|boulevard|place|chemin|allée|impasse)[^,\n]+,\s*\d{5}\s+[A-Za-zéèêàâôùûîïç]+)'
    address_match = re.search(address_pattern, full_text, re.IGNORECASE)
    if address_match:
        invoice_data['client_address'] = address_match.group(1).strip()
    
    # Extract client name
    name_with_label_pattern = r'(?:^|[.!?]\s+)(?:Nom|Pour|Client)\s*:\s*([A-Z][a-zéèêàâôùûîïç]+\s+[A-Z][a-zéèêàâôùûîïç]+)'
    name_match = re.search(name_with_label_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    
    if name_match:
        invoice_data['client_name'] = name_match.group(1).strip()
    elif invoice_data.get('client_address'):
        addr_pos = full_text.find(invoice_data['client_address'])
        if addr_pos > 0:
            before_address = full_text[max(0, addr_pos-100):addr_pos]
            name_near_address = r'([A-Z][a-zéèêàâôùûîïç]+\s+[A-Z][a-zéèêàâôùûîïç]+)[,\s]*$'
            name_match3 = re.search(name_near_address, before_address)
            if name_match3:
                potential_name = name_match3.group(1).strip()
                if potential_name.split()[0] not in ['Votre', 'Donner', 'Merci', 'Complet']:
                    invoice_data['client_name'] = potential_name
    
    if not invoice_data['client_name']:
        name_pattern = r'\b([A-Z][a-zéèêàâôùûîïç]+)\s+([A-Z][a-zéèêàâôùûîïç]+)\b'
        name_matches = re.findall(name_pattern, full_text)
        if name_matches:
            excluded_words = {'Today', 'Bonjour', 'Pour', 'Tel', 'Nom', 'Adresse', 'Prix', 'Total', 
                            'Date', 'Commander', 'Commande', 'Merci', 'Parfait', 'Service',
                            'Votre', 'Donner', 'Pouvez', 'Combien', 'Quelle', 'Couleur'}
            for first, last in name_matches:
                if (first not in excluded_words and last not in excluded_words and 
                    len(first) >= 4 and len(last) >= 4 and 
                    first[0].isupper() and last[0].isupper()):
                    invoice_data['client_name'] = f"{first} {last}"
                    break
    
    if not invoice_data['client_name']:
        invoice_data['client_name'] = "Client"
    
    # Extract iPhone orders
    iphone_pattern = r'(\d+)\s*i[Pp]hone[s]?\s*(\d{0,2})?'
    iphone_matches = re.findall(iphone_pattern, full_text, re.IGNORECASE)
    
    UNIT_PRICE = 250.0
    items_found = []
    total_amount = 0.0
    
    if iphone_matches:
        quantities = {}
        for qty_str, model in iphone_matches:
            try:
                qty = int(qty_str)
                model_num = model if model else ""
                key = f"iPhone {model_num}".strip()
                if key not in quantities:
                    quantities[key] = 0
                quantities[key] += qty
            except ValueError:
                pass
        
        for description, qty in quantities.items():
            item_total = qty * UNIT_PRICE
            item = {
                'description': description,
                'quantity': qty,
                'unit_price': UNIT_PRICE,
                'total': item_total
            }
            items_found.append(item)
            total_amount += item_total
    
    # Look for explicit total
    total_patterns = [
        r'(?:total|prix\s*total|montant\s*total)[:\s]*(\d+(?:[.,]\d{2})?)\s*[€$]',
        r'(\d+(?:[.,]\d{2})?)\s*[€$](?:\s*total)?',
    ]
    
    extracted_total = None
    for pattern in total_patterns:
        total_match = re.search(pattern, full_text, re.IGNORECASE)
        if total_match:
            try:
                extracted_total = float(total_match.group(1).replace(',', '.'))
                break
            except ValueError:
                pass
    
    if extracted_total and items_found:
        total_amount = extracted_total
        total_qty = sum(item['quantity'] for item in items_found)
        expected_total = total_qty * UNIT_PRICE
        if abs(expected_total - extracted_total) > 0.01:
            correct_qty = int(round(extracted_total / UNIT_PRICE))
            if correct_qty > 0 and items_found:
                items_found[0]['quantity'] = correct_qty
                items_found[0]['total'] = extracted_total
    
    # Extract colors
    color_pattern = r'(\d+)?\s*(noir|blanc|rouge|rose|bleu|vert)s?'
    color_matches = re.findall(color_pattern, full_text, re.IGNORECASE)
    
    if color_matches and items_found:
        colors_found = []
        for qty, color in color_matches:
            if qty:
                colors_found.append(f"{qty} {color}")
            else:
                colors_found.append(color)
        if colors_found:
            items_found[0]['description'] += f" ({', '.join(colors_found)})"
    
    if not items_found:
        general_price_pattern = r'(?:prix|price|montant)[:\s]*(\d+(?:[.,]\d{2})?)\s*[€$]'
        price_match = re.search(general_price_pattern, full_text, re.IGNORECASE)
        if price_match:
            try:
                price = float(price_match.group(1).replace(',', '.'))
                items_found.append({
                    'description': 'Service / Produit',
                    'quantity': 1,
                    'unit_price': price,
                    'total': price
                })
                total_amount = price
            except ValueError:
                pass
    
    if not items_found:
        items_found.append({
            'description': 'Service / Prestation',
            'quantity': 1,
            'unit_price': 0.0,
            'total': 0.0
        })
    
    invoice_data['items'] = items_found
    invoice_data['total'] = total_amount
    return invoice_data


def generate_pdf_invoice(invoice_data, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    title = Paragraph("FACTURE / INVOICE", title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    info_data = [
        ['Numéro de facture:', invoice_data['invoice_number']],
        ['Date:', invoice_data['date']],
        ['Client:', invoice_data['client_name']],
    ]
    
    if invoice_data.get('client_address'):
        info_data.append(['Adresse:', invoice_data['client_address']])
    
    info_table = Table(info_data, colWidths=[5*cm, 10*cm])
    info_table.setStyle(TableStyle([
        ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
        ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    items_data = [['Description', 'Quantité', 'Prix Unitaire', 'Total']]
    
    for item in invoice_data['items']:
        items_data.append([
            item['description'],
            str(item['quantity']),
            f"{item['unit_price']:.2f} €",
            f"{item['total']:.2f} €"
        ])
    
    items_data.append(['', '', 'TOTAL:', f"{invoice_data['total']:.2f} €"])
    
    items_table = Table(items_data, colWidths=[8*cm, 3*cm, 3*cm, 3*cm])
    items_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('FONT', (0, 1), (-1, -2), 'Helvetica', 9),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.grey),
        ('FONT', (0, -1), (-1, -1), 'Helvetica-Bold', 11),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('ALIGN', (2, -1), (-1, -1), 'RIGHT'),
        ('TOPPADDING', (0, -1), (-1, -1), 10),
    ]))
    story.append(items_table)
    
    if invoice_data.get('description'):
        story.append(Spacer(1, 20))
        story.append(Paragraph('<b>Notes:</b>', styles['Normal']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(invoice_data['description'][:500], styles['Normal']))
    
    doc.build(story)
    return output_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/check-ollama', methods=['GET'])
def check_ollama():
    return jsonify({
        'available': LLM_AVAILABLE,
        'model': 'llama3.2:1b' if LLM_AVAILABLE else None
    })


@app.route('/api/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Type de fichier non autorisé'}), 400
        
        method = request.form.get('method', 'regex')
        
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
        
        # Try Ollama if selected, fallback to regex
        if method == 'ollama' and LLM_AVAILABLE:
            invoice_data, error = interpret_with_ollama(extracted_text)
            if error:
                invoice_data = interpret_invoice_data(extracted_text)
                method_used = 'regex'
            else:
                method_used = 'ollama'
        else:
            invoice_data = interpret_invoice_data(extracted_text)
            method_used = 'regex'
        
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
    print("🧾 OCR Invoice Generator - Starting...")
    print("=" * 60)
    print(f"✓ OCR Engine: {OCR_ENGINE if OCR_ENGINE else 'Not available'}")
    print(f"✓ LLM: {'Ollama (optional)' if LLM_AVAILABLE else 'Not available (regex only)'}")
    print("=" * 60)
    print("🌐 Server running at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

