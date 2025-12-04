import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader

    PDF_AVAILABLE = True
except ImportError:
    logger.error("PyPDF2 not installed. Run: pip install PyPDF2")
    PDF_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    logger.warning("Gemini not available. Install: pip install google-generativeai")
    GEMINI_AVAILABLE = False


class GeminiHelper:
    """Enhanced Gemini initialization with proper model detection"""

    def __init__(self, api_key: str):
        """Gemini API configure"""
        self.api_key = api_key
        self.model = None
        self.model_name = None
        self.available = False

        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai package not installed")
            return

        if not api_key:
            logger.warning("No API key provided")
            return

        self._initialize()

    def _initialize(self):
        """Initialize Gemini with smart model detection"""
        try:
            genai.configure(api_key=self.api_key)
            logger.info("Scanning available Gemini models...")

            # Get all available models
            available_models = []
            try:
                for model in genai.list_models():
                    if 'generateContent' in model.supported_generation_methods:
                        model_name = model.name.replace('models/', '')
                        available_models.append({
                            'name': model_name,
                            'display_name': model.display_name,
                            'full_name': model.name
                        })
                        logger.info(f"  ✓ Found: {model_name}")
            except Exception as e:
                logger.warning(f"Could not list models: {e}")
                # Fallback to common models
                available_models = [
                    {'name': 'gemini-1.5-flash', 'display_name': 'Gemini 1.5 Flash'},
                    {'name': 'gemini-1.5-pro', 'display_name': 'Gemini 1.5 Pro'},
                ]

            # Priority order (newest/best first)
            preferred_order = [
                'gemini-2.0-flash-exp',
                'gemini-exp-1206',
                'gemini-2.0-flash-thinking-exp',
                'gemini-exp-1121',
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash-002',
                'gemini-1.5-flash-8b',
                'gemini-1.5-flash',
                'gemini-1.5-pro-latest',
                'gemini-1.5-pro-002',
                'gemini-1.5-pro',
            ]

            # Try models in order of preference
            for preferred in preferred_order:
                for available in available_models:
                    if preferred == available['name'] or preferred in available['name']:
                        if self._test_model(available['name']):
                            self.model_name = available['name']
                            self.available = True
                            logger.info(f"\nSuccessfully initialized: {self.model_name}\n")
                            return

            # If no preferred model worked, try first available
            if available_models:
                for model_info in available_models:
                    if self._test_model(model_info['name']):
                        self.model_name = model_info['name']
                        self.available = True
                        logger.info(f"\nSuccessfully initialized: {self.model_name}\n")
                        return

            logger.error("No working Gemini model found")
            self._print_troubleshooting()

        except Exception as e:
            logger.error(f"Gemini initialization failed: {e}")
            self._print_troubleshooting()

    def _test_model(self, model_name: str) -> bool:
        """Test if a model works"""
        try:
            logger.info(f"Testing model: {model_name}...")
            test_model = genai.GenerativeModel(model_name)
            response = test_model.generate_content(
                "Say 'ready'",
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 10
                }
            )

            if response and response.text:
                logger.info(f"{model_name} is working!")
                self.model = test_model
                return True
            else:
                logger.warning(f"{model_name} returned empty response")
                return False

        except Exception as e:
            logger.warning(f"{model_name} failed: {str(e)[:100]}")
            return False

    def _print_troubleshooting(self):
        """Print troubleshooting info"""
        logger.info("\n" + "=" * 80)
        logger.info("TROUBLESHOOTING STEPS:")
        logger.info("=" * 80)
        logger.info("1. Verify API key at: https://aistudio.google.com/apikey")
        logger.info("2. Update package: pip install --upgrade google-generativeai")
        logger.info("3. Check if you have API access enabled")
        logger.info("4. Try generating content manually at: https://aistudio.google.com")
        logger.info("5. Ensure you're not hitting rate limits")
        logger.info("=" * 80 + "\n")

    def generate(self, prompt: str, max_tokens: int = 1500) -> Optional[str]:
        # AI response generate and text return
        """Generate content with error handling"""
        if not self.available or not self.model:
            return None

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': max_tokens,
                    'top_p': 0.95,
                    'top_k': 40
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )

            if response and response.text:
                return response.text
            else:
                logger.warning("Empty response from Gemini")
                return None

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None


class SmartPDFSummarizer:
    """Intelligent PDF analysis - AI first, then rule-based fallback"""

    def __init__(self, api_key: str = None):
        self.gemini = None

        if api_key:
            self.gemini = GeminiHelper(api_key)

    def extract_full_text(self, pdf_path: str) -> Tuple[str, str]:
        """Extract complete text from PDF"""
        if not PDF_AVAILABLE:
            raise RuntimeError("PyPDF2 not installed")

        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)

                # Extract all text
                text_parts = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.debug(f"Page {i + 1} extraction failed: {e}")

                full_text = '\n'.join(text_parts)
                return full_text, os.path.basename(pdf_path)

        except Exception as e:
            logger.error(f"Failed to read {pdf_path}: {e}")
            return "", os.path.basename(pdf_path)

    def extract_entities(self, text: str) -> Dict:
        """Extract key entities from text"""
        entities = {}

        # Promoter/Developer
        promoter_patterns = [
            r'(?:promoter|developer)[:\s]+([A-Z][A-Za-z\s&\-\.LLP]{3,60})(?:\s+having\s+PAN|the\s+auditor)',
            r'of\s+([A-Z][A-Za-z\s&\-\.LLP]{3,60})\s+having\s+PAN',
            r'([A-Z][A-Z\s&\-\.LLP]{5,60})\s+\(Promoter',
        ]
        for pattern in promoter_patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                entities['promoter'] = match.group(1).strip()
                break

        # Location (city)
        location_match = re.search(
            r'\b(ahmedabad|mumbai|delhi|bangalore|bengaluru|pune|hyderabad|chennai|kolkata|jaipur|surat|gurgaon|noida|gujarat)\b',
            text[:3000], re.IGNORECASE
        )
        if location_match:
            entities['location'] = location_match.group(1).title()

        # Financial Year
        fy_patterns = [
            r'year\s+ending\s+on\s+(\d{2}/\d{2}/\d{4})',
            r'(?:FY|financial\s+year)[:\s]*(\d{4}[-/]?\d{2,4})',
            r'for\s+year\s+(\d{4})',
        ]
        for pattern in fy_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                entities['financial_year'] = match.group(1)
                break

        # Bank Account Number
        bank_account_patterns = [
            r'Account\s+Number\s*[:=\s]+(\d{10,20})',
            r'A/C\s+No[:\s]+(\d{10,20})',
            r'Account\s+No[:\s]+(\d{10,20})',
        ]
        for pattern in bank_account_patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                entities['bank_account'] = match.group(1)
                break

        # Bank Name
        bank_name_patterns = [
            r'Bank\s+Name\s*[:=\s]+([A-Z][A-Za-z\s&]+?)(?:\n|Branch)',
            r'Bank[:\s]+([A-Z][A-Za-z\s&]+Bank)',
        ]
        for pattern in bank_name_patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                entities['bank_name'] = match.group(1).strip()
                break

        return entities

    def extract_key_points_rule_based(self, text: str) -> List[str]:
        """Extract important points using rule-based approach"""

        # Keywords for RERA/Compliance documents
        keywords = [
            'rera', 'registration', 'compliance', 'bank account',
            'withdrawal', 'deposit', 'allottee', 'form', 'approval',
            'certificate', 'authority', 'audit', 'examined', 'confirmed'
        ]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Score sentences
        scored = []
        for sent in sentences:
            words = sent.split()
            if len(words) < 8 or len(words) > 120:
                continue

            # Skip all-caps headers
            if sent.isupper() and len(words) < 15:
                continue

            score = 0
            sent_lower = sent.lower()

            # Keyword matching
            for keyword in keywords:
                if keyword in sent_lower:
                    score += 5

            # Numbers boost (amounts, dates)
            if re.search(r'\d+', sent):
                score += 3

            # Important phrases
            important_phrases = [
                'shall', 'must', 'required to', 'hereby', 'certified',
                'confirm', 'examined', 'whether', 'approval', 'registered',
                'yes', 'no', 'withdrawn', 'deposited'
            ]
            for phrase in important_phrases:
                if phrase in sent_lower:
                    score += 4

            if score > 0:
                scored.append((score, sent.strip()))

        # Sort by score
        scored.sort(reverse=True, key=lambda x: x[0])

        # Select diverse points
        selected = []
        for score, sent in scored:
            if len(selected) >= 10:
                break

            # Check uniqueness
            is_unique = True
            for existing in selected:
                if self._similarity(sent, existing) > 0.65:
                    is_unique = False
                    break

            if is_unique:
                clean = self._clean_sentence(sent)
                if clean and len(clean) > 30:
                    selected.append(clean)

        return selected if selected else ["No significant points identified."]

    def _clean_sentence(self, sent: str) -> str:
        """Clean and format sentence"""
        # Remove extra whitespace
        sent = re.sub(r'\s+', ' ', sent).strip()

        # Remove leading numbers/bullets
        sent = re.sub(r'^[\d\)\.\(\]]+[\.\):\s]+', '', sent)
        sent = re.sub(r'^[•\-\*→▪]+\s*', '', sent)

        # Capitalize
        if sent and sent[0].islower():
            sent = sent[0].upper() + sent[1:]

        # Add period
        if sent and sent[-1] not in '.!?':
            sent += '.'

        # Length limit
        if len(sent) > 500:
            sent = sent[:497] + '...'

        return sent.strip()

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity"""
        w1 = set(s1.lower().split())
        w2 = set(s2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def generate_ai_summary(self, text: str, entities: Dict) -> Optional[Dict]:
        """Generate summary using Gemini AI"""
        if not self.gemini or not self.gemini.available:
            return None

        entities_str = json.dumps(entities, indent=2, ensure_ascii=False)

        prompt = f"""Analyze this RERA/Compliance document and provide a comprehensive summary.

KEY ENTITIES FOUND:
{entities_str}

DOCUMENT TEXT (first 5000 characters):
{text[:5000]}

Please provide:

1. SUMMARY (3-4 sentences):
   - Main purpose of this document
   - Key project/company details
   - Important compliance status or findings

2. KEY POINTS (10 specific findings):
   - Compliance confirmations (YES/NO answers)
   - Financial details and amounts
   - Bank account information
   - Deadlines and dates mentioned
   - Approval statuses
   - Audit confirmations
   - Any violations or issues
   - Important certifications or requirements

Format your response EXACTLY as:

SUMMARY:
[Your 3-4 sentence summary here]

KEY POINTS:
1. [Point 1 - be specific with numbers/dates]
2. [Point 2]
3. [Point 3]
4. [Point 4]
5. [Point 5]
6. [Point 6]
7. [Point 7]
8. [Point 8]
9. [Point 9]
10. [Point 10]

Each point must be specific, actionable, and include relevant details."""

        response_text = self.gemini.generate(prompt, max_tokens=1500)

        if response_text:
            return self._parse_ai_response(response_text)

        return None

    def _parse_ai_response(self, text: str) -> Dict:
        """Parse AI response"""
        lines = text.strip().split('\n')
        summary = ""
        key_points = []
        mode = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if 'SUMMARY:' in line.upper():
                summary = line.split(':', 1)[1].strip() if ':' in line else ""
                mode = 'summary'
            elif 'KEY POINTS:' in line.upper() or 'KEY FINDINGS:' in line.upper():
                mode = 'points'
            elif mode == 'summary' and not line.startswith(('1.', '2.', '3.')):
                summary += ' ' + line
            elif mode == 'points':
                # Extract numbered points
                match = re.match(r'^(\d+[\.\)])\s*(.+)$', line)
                if match:
                    key_points.append(match.group(2).strip())
                elif line.startswith(('-', '•', '*')):
                    key_points.append(line.lstrip('-•* ').strip())

        return {
            'summary': summary.strip(),
            'key_points': key_points[:10]
        }

    def _generate_rule_based_summary(self, text: str, entities: Dict) -> str:
        """Generate simple rule-based summary"""
        parts = ["Document analysis"]

        if 'promoter' in entities:
            parts.append(f"for {entities['promoter']}")

        if 'location' in entities:
            parts.append(f"in {entities['location']}")

        if 'financial_year' in entities:
            parts.append(f"for FY {entities['financial_year']}")

        summary = ' '.join(parts) + '.'

        # Add first meaningful sentence
        sentences = re.split(r'[.!?]\s+', text[:2500])
        for sent in sentences:
            if 12 <= len(sent.split()) <= 35 and not sent.isupper():
                clean = self._clean_sentence(sent)
                if clean and len(summary) + len(clean) < 500:
                    summary += ' ' + clean
                    break

        return summary

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """Analyze PDF - AI first, then rule-based fallback"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        logger.info(f"{'=' * 80}")

        # Extract text
        full_text, filename = self.extract_full_text(pdf_path)

        if not full_text or len(full_text) < 100:
            logger.error("Insufficient text extracted")
            return {
                'filename': filename,
                'summary': 'Error: Could not extract text from PDF',
                'key_points': [],
                'promoter': None,
                'location': None,
                'financial_year': None,
                'bank_account': None,
                'bank_name': None
            }

        # Extract entities
        entities = self.extract_entities(full_text)
        logger.info(f"Entities found: {len(entities)}")

        # Try AI first (ONLY IF AVAILABLE)
        if self.gemini and self.gemini.available:
            logger.info(f"Using AI: {self.gemini.model_name}")
            ai_result = self.generate_ai_summary(full_text, entities)

            if ai_result and ai_result.get('summary') and len(ai_result.get('key_points', [])) >= 5:
                logger.info(f"AI analysis successful")
                return {
                    'filename': filename,
                    'summary': ai_result['summary'],
                    'key_points': ai_result['key_points'],
                    'promoter': entities.get('promoter'),
                    'location': entities.get('location'),
                    'financial_year': entities.get('financial_year'),
                    'bank_account': entities.get('bank_account'),
                    'bank_name': entities.get('bank_name')
                }
            else:
                logger.warning("AI analysis incomplete, falling back to rules")

        # Rule-based fallback
        logger.info("Using rule-based extraction")
        summary = self._generate_rule_based_summary(full_text, entities)
        key_points = self.extract_key_points_rule_based(full_text)

        logger.info(f"Rule-based analysis complete")

        return {
            'filename': filename,
            'summary': summary,
            'key_points': key_points,
            'promoter': entities.get('promoter'),
            'location': entities.get('location'),
            'financial_year': entities.get('financial_year'),
            'bank_account': entities.get('bank_account'),
            'bank_name': entities.get('bank_name')
        }

    def analyze_folder(self, folder_path: str) -> List[Dict]:
        """Analyze all PDFs in folder"""
        folder = Path(folder_path)

        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []

        pdf_files = sorted(folder.glob('*.pdf'))
        if not pdf_files:
            logger.warning(f"No PDFs found in {folder_path}")
            return []

        logger.info(f"\nFound {len(pdf_files)} PDF files\n")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}]")
            result = self.analyze_pdf(str(pdf_path))
            results.append(result)

        return results


def save_results(results: List[Dict], output_file: str):
    """Save to JSON with clean format"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved: {output_file}")


def print_report(results: List[Dict]):
    """Print beautiful report"""
    print("\n" + "=" * 100)
    print("PDF ANALYSIS RESULTS".center(100))
    print("=" * 100)
    print(f"\nTotal Files: {len(results)}\n")

    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 100}")
        print(f"[{i}] {r['filename']}")
        print(f"{'=' * 100}")

        # Summary
        print(f"\nSUMMARY:")
        print(f"{r['summary']}")

        # Entities
        print(f"\nKEY DETAILS:")
        print(f" • Promoter: {r.get('promoter') or 'Not found'}")
        print(f" • Location: {r.get('location') or 'Not found'}")
        print(f" • Financial Year: {r.get('financial_year') or 'Not found'}")
        print(f" • Bank Account: {r.get('bank_account') or 'Not found'}")
        print(f" • Bank Name: {r.get('bank_name') or 'Not found'}")

        # Key Points
        print(f"\nKEY POINTS ({len(r['key_points'])}):")
        print("-" * 100)
        for j, point in enumerate(r['key_points'], 1):
            print(f"\n{j}. {point}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!".center(100))
    print("=" * 100 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Smart PDF Analyzer - AI First with Rule-Based Fallback',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--folder', '-f', required=True, help='Folder with PDFs')
    parser.add_argument('--output', '-o', default='pdf_results.json', help='Output JSON file')
    parser.add_argument('--apikey', '-k', default=None, help='Gemini API key (optional)')

    args = parser.parse_args()

    if not PDF_AVAILABLE:
        print("PyPDF2 not installed. Run: pip install PyPDF2")
        return 1

    print("\n" + "=" * 100)
    print("SMART PDF ANALYZER - AI FIRST WITH FALLBACK".center(100))
    print("=" * 100)
    print(f"Folder: {args.folder}")
    print(f"Output: {args.output}")
    print(f"AI: {'Enabled (will try AI first)' if args.apikey else 'Disabled (rule-based only)'}")
    print("=" * 100)

    # Process PDFs
    summarizer = SmartPDFSummarizer(api_key=args.apikey)
    results = summarizer.analyze_folder(args.folder)

    if not results:
        print("\nNo results generated")
        return 1

    # Save and display
    save_results(results, args.output)
    print_report(results)

    print("\nTIP: If AI failed, it automatically used rule-based extraction.")
    print("   Check the logs above to see which method was used for each file.\n")

    return 0


if __name__ == '__main__':
    exit(main())