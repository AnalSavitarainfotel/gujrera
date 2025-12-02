import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from PyPDF2 import PdfReader

    PDF_AVAILABLE = True
except ImportError:
    logger.error("PyPDF2 not installed. Install: pip install PyPDF2")
    PDF_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class ComplianceExtractor:
    """Advanced compliance point extraction with best logic"""

    COMPLIANCE_PATTERNS = {
        'mandatory_requirements': {
            'weight': 10,
            'patterns': [
                r'(?:shall|must|required to|mandatory|obligated to)\s+[\w\s]{5,80}',
                r'promoter\s+shall\s+[\w\s]{5,60}',
                r'developer\s+(?:shall|must)\s+[\w\s]{5,60}',
                r'(?:requirement|obligation)\s+(?:is|to|for)\s+[\w\s]{5,50}',
            ]
        },
        'financial_terms': {
            'weight': 10,
            'patterns': [
                r'(?:rs\.?|rupees|inr)\s*\d+[,\d]*(?:\.\d+)?[\w\s]{0,40}',
                r'\d+\s*(?:%|percent|percentage)\s+(?:of|as|for)[\w\s]{0,40}',
                r'(?:deposit|payment|fee|penalty|fine)\s+of\s+(?:rs\.?|rupees)?\s*\d+[\w\s]{0,40}',
                r'bank\s+(?:guarantee|account)\s+[\w\s]{5,50}',
                r'escrow\s+account[\w\s]{5,50}',
            ]
        },
        'deadlines': {
            'weight': 10,
            'patterns': [
                r'within\s+\d+\s+(?:days|months|years|weeks)[\w\s]{0,50}',
                r'(?:quarterly|monthly|annually|half-yearly)\s+(?:basis|submission)[\w\s]{0,40}',
                r'deadline\s+(?:is|of|for)[\w\s]{5,50}',
                r'due\s+(?:date|on|by)[\w\s]{5,40}',
            ]
        },
        'documentation': {
            'weight': 9,
            'patterns': [
                r'(?:maintain|keep|preserve)\s+(?:records|documents|books)[\w\s]{0,50}',
                r'(?:certificate|form|register)\s+[\w\s]{5,50}',
                r'upload(?:ing)?\s+(?:of|on)\s+(?:website|portal)[\w\s]{0,40}',
            ]
        },
        'submission_filing': {
            'weight': 9,
            'patterns': [
                r'(?:submit|file|upload)\s+[\w\s]{5,60}(?:to|with|on)[\w\s]{0,40}',
                r'form\s+[A-Z0-9\-]+[\w\s]{0,50}',
                r'submission\s+of\s+[\w\s]{5,50}',
            ]
        },
        'prohibitions': {
            'weight': 10,
            'patterns': [
                r'(?:shall\s+not|not\s+allowed|prohibited|forbidden)[\w\s]{5,60}',
                r'(?:cannot|must\s+not)\s+[\w\s]{5,50}(?:without|unless)',
                r'no\s+(?:person|promoter|developer)\s+shall[\w\s]{5,50}',
            ]
        },
        'penalties': {
            'weight': 9,
            'patterns': [
                r'penalty\s+(?:of|upto|not\s+exceeding)\s+[\w\s]{5,60}',
                r'fine\s+(?:of|upto)[\w\s]{5,50}',
                r'(?:violation|breach|non-compliance)[\w\s]{5,60}',
            ]
        },
        'verification': {
            'weight': 8,
            'patterns': [
                r'(?:verification|verify|audit)\s+[\w\s]{5,50}',
                r'certified\s+by[\w\s]{5,50}',
                r'(?:inspect|examination)\s+[\w\s]{5,50}',
            ]
        },
        'rera_specific': {
            'weight': 10,
            'patterns': [
                r'rera\s+[\w\s]{5,60}',
                r'registration\s+(?:number|certificate)[\w\s]{5,50}',
                r'real\s+estate\s+regulatory\s+authority[\w\s]{0,50}',
            ]
        },
        'stakeholder_rights': {
            'weight': 8,
            'patterns': [
                r'(?:allottee|buyer)\s+(?:shall|entitled|has\s+right)[\w\s]{5,60}',
                r'possession\s+[\w\s]{5,50}',
                r'(?:rights|obligations)\s+of\s+[\w\s]{5,50}',
            ]
        }
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove numbering and bullets
        text = re.sub(r'^[\d\)\.\(\]]+[\.\):\s]+', '', text)
        text = re.sub(r'^[•\-\*→▪]+\s*', '', text)

        # Fix spacing
        text = re.sub(r'\s+([,\.;:])', r'\1', text)

        # Capitalize
        if text and text[0].islower():
            text = text[0].upper() + text[1:]

        # Add period
        if text and text[-1] not in '.!?':
            text += '.'

        # Limit length
        if len(text) > 500:
            text = text[:497] + '...'

        return text.strip()

    @classmethod
    def similarity(cls, t1: str, t2: str) -> float:
        """Calculate Jaccard similarity"""
        w1, w2 = set(t1.lower().split()), set(t2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    @classmethod
    def score_sentence(cls, sentence: str) -> Tuple[int, Set[str]]:
        """Score sentence based on compliance patterns"""
        text_lower = sentence.lower()
        score = 0
        categories = set()

        words = sentence.split()
        if len(words) < 10 or len(words) > 100:
            return 0, set()

        # Skip headers and noise
        if sentence.isupper() and len(words) < 25:
            return 0, set()
        if re.match(r'^[\d\s\-/,.:()]+$', sentence):
            return 0, set()

        # Score patterns
        for category, config in cls.COMPLIANCE_PATTERNS.items():
            cat_score = 0
            for pattern in config['patterns']:
                if re.search(pattern, text_lower):
                    cat_score += 5

            if cat_score > 0:
                score += cat_score * config['weight']
                categories.add(category)

        # Bonus for specific indicators
        if re.search(r'(?:section|clause|rule)\s+\d+', text_lower):
            score += 8
        if re.search(r'form\s+[A-Z0-9]+', text_lower):
            score += 7
        if re.search(r'\d{4}\s+(?:act|rules)', text_lower):
            score += 6

        # Action words bonus
        action_words = ['shall', 'must', 'required', 'mandatory', 'ensure',
                        'submit', 'maintain', 'verify', 'comply']
        for word in action_words:
            if re.search(r'\b' + word + r'\b', text_lower):
                score += 3

        return score, categories

    @classmethod
    def extract_points(cls, text: str, target: int = 10) -> List[str]:
        """Extract diverse compliance points"""
        if not text or len(text) < 200:
            return ["No sufficient text content for analysis"]

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Score all sentences
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if 80 <= len(sent) <= 1000:
                score, cats = cls.score_sentence(sent)
                if score > 0:
                    scored.append({'text': sent, 'score': score, 'categories': cats})

        # Sort by score
        scored.sort(key=lambda x: x['score'], reverse=True)

        # Select diverse points
        selected = []
        used_categories = defaultdict(int)

        for item in scored:
            if len(selected) >= target:
                break

            cleaned = cls.clean_text(item['text'])
            if not cleaned or len(cleaned.split()) < 10:
                continue

            # Check diversity
            is_diverse = True
            for existing in selected:
                if cls.similarity(cleaned, existing) > 0.55:
                    is_diverse = False
                    break

            if not is_diverse:
                continue

            # Check category balance
            category_bonus = sum(1 for cat in item['categories'] if used_categories[cat] < 2)

            # Accept if diverse and helps coverage
            if is_diverse and (category_bonus > 0 or len(selected) < target // 2):
                selected.append(cleaned)
                for cat in item['categories']:
                    used_categories[cat] += 1

        # Fallback: numbered/bulleted points
        if len(selected) < target // 2:
            structured = cls.extract_structured(text, target - len(selected))
            for point in structured:
                if len(selected) >= target:
                    break
                if point not in selected and all(cls.similarity(point, s) < 0.6 for s in selected):
                    selected.append(point)

        # Remove duplicates
        unique = []
        seen = set()
        for point in selected:
            norm = point.lower().strip()
            if norm not in seen:
                unique.append(point)
                seen.add(norm)

        return unique[:target] if unique else ["No compliance points identified - manual review required"]

    @classmethod
    def extract_structured(cls, text: str, count: int) -> List[str]:
        """Extract numbered/bulleted points"""
        points = []
        patterns = [
            r'(?:^|\n)\s*\d+\.\s+([^\n]{80,800})',
            r'(?:^|\n)\s*\d+\)\s+([^\n]{80,800})',
            r'(?:^|\n)\s*[•\-\*→]\s+([^\n]{80,800})',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for content in matches:
                if len(points) >= count:
                    break
                cleaned = cls.clean_text(content)
                if cleaned and len(cleaned.split()) >= 10:
                    points.append(cleaned)

        return points


class DescriptionExtractor:
    """Extract document description"""

    @classmethod
    def extract(cls, text: str, filename: str) -> str:
        """Generate description from text and filename"""
        # Detect document type
        doc_type = cls.detect_type(text, filename)

        # Extract key entities
        entities = {
            'project': re.search(
                r'(?:project|name\s+of\s+project)[:\s]+([A-Z][A-Za-z\s&\-]{3,50})',
                text[:2000], re.IGNORECASE
            ),
            'registration': re.search(
                r'(?:registration|rera|reg\.?)\s*(?:no\.?|number|#)[:\s]*([A-Z0-9/\-]+)',
                text[:2000], re.IGNORECASE
            ),
            'year': re.search(r'\b(20\d{2})\b', text[:1500]),
            'location': re.search(
                r'\b(ahmedabad|mumbai|delhi|bangalore|pune|hyderabad|chennai|kolkata|jaipur|surat)\b',
                text[:2000], re.IGNORECASE
            ),
            'promoter': re.search(
                r'(?:promoter|developer)[:\s]+([A-Z][A-Za-z\s&\-\.]{3,40})',
                text[:2000], re.IGNORECASE
            ),
        }

        # Build description
        parts = [doc_type]

        if entities['project']:
            parts.append(f"for project '{entities['project'].group(1).strip()}'")

        if entities['promoter']:
            parts.append(f"by {entities['promoter'].group(1).strip()}")

        if entities['registration']:
            parts.append(f"(Registration: {entities['registration'].group(1).strip()})")

        if entities['location']:
            parts.append(f"in {entities['location'].group(1).title()}")

        if entities['year']:
            parts.append(f"for year {entities['year'].group(1)}")

        description = ' '.join(parts)

        # Add context sentence
        sentences = re.split(r'[.!?]\s+', text[:2500])
        for sent in sentences:
            words = sent.split()
            if 15 <= len(words) <= 45 and not sent.isupper():
                cleaned = ComplianceExtractor.clean_text(sent)
                if cleaned and len(description) + len(cleaned) < 500:
                    description += '. ' + cleaned
                    break

        return description[:600]

    @classmethod
    def detect_type(cls, text: str, filename: str) -> str:
        """Detect document type"""
        text_sample = (filename + ' ' + text[:3000]).lower()

        patterns = {
            'RERA Compliance Certificate': [r'rera\s+compliance', r'form\s*[345]'],
            'RERA Registration Certificate': [r'rera\s+registration', r'registration\s+certificate'],
            'Audit Report': [r'audit(?:or)?(?:\'s)?\s+report', r'independent\s+auditor'],
            'Financial Statement': [r'balance\s+sheet', r'profit\s+(?:and|&)\s+loss'],
            'Quarterly Return': [r'quarterly\s+return', r'quarter\s+ended'],
            'Annual Return': [r'annual\s+return', r'financial\s+year'],
            'Circular/Notification': [r'circular', r'notification'],
            'Compliance Report': [r'compliance\s+report'],
        }

        scores = defaultdict(int)
        for doc_type, pats in patterns.items():
            for pat in pats:
                scores[doc_type] += len(re.findall(pat, text_sample))

        return max(scores, key=scores.get) if scores else 'Compliance Document'


class GeminiAnalyzer:
    """Optional Gemini AI analysis"""

    def __init__(self, api_key: str = None):
        self.available = False
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True
                logger.info("Gemini AI enabled")
            except:
                logger.warning("Gemini AI initialization failed")

    def analyze(self, text: str, filename: str) -> Dict:
        """AI-powered analysis"""
        if not self.available:
            return None

        prompt = f"""Analyze this compliance document and extract:

1. DESCRIPTION (2-3 sentences): Project name, registration number, location, purpose
2. KEY COMPLIANCE POINTS (10 unique points):
   - Financial obligations, deadlines, documentation requirements
   - Submission procedures, prohibitions, penalties
   - Verification requirements, stakeholder rights

Each point must be specific, actionable, and different.

Document: {filename}
Text: {text[:5000]}

Format:
DESCRIPTION: [description]
POINTS:
- [point 1]
- [point 2]
...
- [point 10]"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2, 'max_output_tokens': 1200}
            )
            return self._parse(response.text)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None

    def _parse(self, text: str) -> Dict:
        """Parse AI response"""
        lines = text.strip().split('\n')
        description = ""
        points = []
        mode = None

        for line in lines:
            line = line.strip()
            if line.upper().startswith('DESCRIPTION:'):
                description = line.split(':', 1)[1].strip()
                mode = 'desc'
            elif 'POINTS:' in line.upper():
                mode = 'points'
            elif mode == 'desc' and line and not line.startswith('-'):
                description += ' ' + line
            elif mode == 'points' and line.startswith(('-', '•', '*')):
                point = line.lstrip('-•* ').strip()
                if len(point) > 20:
                    points.append(point)

        return {'description': description, 'compliance_points': points}


class PDFProcessor:
    """Main PDF processing class"""

    def __init__(self, api_key: str = None):
        if not PDF_AVAILABLE:
            raise RuntimeError("PyPDF2 required. Install: pip install PyPDF2")
        self.ai = GeminiAnalyzer(api_key)

    def extract_text(self, pdf_path: str, max_pages: int = 50) -> str:
        """Extract text from PDF"""
        text_chunks = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                num_pages = min(len(reader.pages), max_pages)
                for i in range(num_pages):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            text_chunks.append(page_text)
                    except Exception as e:
                        logger.warning(f"Page {i + 1} extraction failed: {e}")
            return '\n'.join(text_chunks).strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ""

    def analyze_pdf(self, pdf_path: str, target_points: int = 10) -> Dict:
        """Analyze single PDF - returns only 3 fields"""
        filename = os.path.basename(pdf_path)
        logger.info(f"Processing: {filename}")

        # Extract text
        text = self.extract_text(pdf_path)

        if not text or len(text) < 200:
            return {
                'filename': filename,
                'description': 'Error: No readable text found in PDF',
                'compliance_points': []
            }

        # Try AI first
        ai_result = self.ai.analyze(text, filename)

        if ai_result and ai_result.get('compliance_points') and len(ai_result['compliance_points']) >= 8:
            description = ai_result['description']
            points = ai_result['compliance_points'][:target_points]
        else:
            # Fallback to rule-based
            description = DescriptionExtractor.extract(text, filename)
            points = ComplianceExtractor.extract_points(text, target_points)

        return {
            'filename': filename,
            'description': description,
            'compliance_points': points
        }

    def analyze_folder(self, folder_path: str, target_points: int = 10) -> List[Dict]:
        """Analyze all PDFs in folder"""
        folder = Path(folder_path)

        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return []

        pdf_files = sorted(folder.glob('*.pdf'))
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files\n")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}]")
            result = self.analyze_pdf(str(pdf_path), target_points)
            results.append(result)

        return results


def save_json(results: List[Dict], output_file: str):
    """Save results to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


def print_results(results: List[Dict]):
    """Print results"""
    print("\n" + "=" * 100)
    print("COMPLIANCE ANALYSIS RESULTS")
    print("=" * 100)
    print(f"Total Files Processed: {len(results)}\n")

    for i, result in enumerate(results, 1):
        print(f"\n{'=' * 100}")
        print(f"[{i}] {result['filename']}")
        print(f"{'=' * 100}")
        print(f"\nDESCRIPTION:")
        print(f"{result['description']}")
        print(f"\nCOMPLIANCE POINTS ({len(result['compliance_points'])} found):")
        print("-" * 100)

        for j, point in enumerate(result['compliance_points'], 1):
            print(f"\n{j}. {point}")

        print("\n")


def main():
    parser = argparse.ArgumentParser(description='PDF Compliance Analyzer - 3 Fields Output')
    parser.add_argument('--folder', '-f', required=True, help='Folder with PDF files')
    parser.add_argument('--output', '-o', default='compliance_results.json', help='Output JSON file')
    parser.add_argument('--points', '-p', type=int, default=10, help='Target points per PDF (default: 10)')
    parser.add_argument('--apikey', '-k', default=None, help='Gemini API key (optional, for better results)')

    args = parser.parse_args()

    if not PDF_AVAILABLE:
        print("ERROR: PyPDF2 not installed. Run: pip install PyPDF2")
        return 1

    print("\n" + "=" * 100)
    print("PDF COMPLIANCE ANALYZER - CLEAN 3-FIELD OUTPUT")
    print("=" * 100)
    print(f"Folder: {args.folder}")
    print(f"Output: {args.output}")
    print(f"Target Points: {args.points}")
    print(f"AI Enhancement: {'Enabled (Gemini)' if args.apikey else 'Disabled (rule-based only)'}")
    print("=" * 100 + "\n")

    # Process
    processor = PDFProcessor(api_key=args.apikey)
    results = processor.analyze_folder(args.folder, args.points)

    if not results:
        print("No results generated.")
        return 1

    # Save and display
    save_json(results, args.output)
    print_results(results)

    print("=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())

# python today.py -f ./downloads -o output.json
# python today_1.py -f ./downloads -o analysis.json  -k AIzaSyBXCPAlNb2B6qlQXkdgfPTsMXWYtGSZasA