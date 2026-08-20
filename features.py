"""
Enhanced 30-Dimension Cybersecurity Feature Extractor for Malicious URL Detection
PhishGuard 2.0 - Standalone Cloud / Google Colab T4 GPU Acceleration Suite
"""

import re
import math
from urllib.parse import urlparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# High-risk disposable/phishing TLD dictionary with risk weights
HIGH_RISK_TLDS = {
    '.cc': 0.85,
    '.xyz': 0.90,
    '.top': 0.90,
    '.ru': 0.70,
    '.cn': 0.70,
    '.tk': 0.95,
    '.ml': 0.95,
    '.ga': 0.95,
    '.cf': 0.95,
    '.gq': 0.95,
    '.click': 0.85,
    '.club': 0.75,
    '.work': 0.80,
    '.link': 0.75,
    '.buzz': 0.85,
    '.fit': 0.80,
    '.info': 0.65,
    '.cam': 0.85,
    '.icu': 0.90,
    '.monster': 0.85,
    '.live': 0.60
}

# Top high-value brand names frequently targeted in phishing attacks
TOP_BRANDS = [
    'paypal', 'microsoft', 'google', 'apple', 'amazon', 'netflix',
    'chase', 'facebook', 'instagram', 'wellsfargo', 'bankofamerica',
    'binance', 'coinbase', 'ebay', 'walmart', 'citibank', 'usps',
    'dhl', 'fedex', 'dropbox', 'adobe', 'linkedin', 'twitter',
    'yahoo', 'roblox', 'steam', 'metamask', 'barclays', 'hsbc',
    'github', 'gitlab', 'wikipedia', 'stackoverflow', 'reddit',
    'spotify', 'youtube', 'zoom', 'salesforce', 'cloudflare'
]
TOP_BRANDS_SET = set(TOP_BRANDS)

# Sensitive phishing intent keywords
PHISHING_KEYWORDS = [
    'login', 'signin', 'verify', 'verification', 'bank', 'account',
    'update', 'secure', 'security', 'wallet', 'confirm', 'confirmation',
    'auth', 'authenticate', 'support', 'alert', 'billing', 'service',
    'recovery', 'portal', 'validation', 'validate', 'password', 'passcode',
    'suspend', 'suspended', 'unlock', 'unusual', 'activity', 'id-verify'
]

SPECIAL_CHARS = ['-', '_', '.', '@', '?', '=', '&', '%', '+', '!', '~', '$', '#', ';', ':']

# Pre-compiled regular expressions for ultra-fast matching
RE_IP = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
RE_HOST_TOKENS = re.compile(r'[-._0-9]+')
RE_WORDS = re.compile(r'[a-zA-Z]+')


def calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon Entropy (randomness / information density) of a string."""
    if not text:
        return 0.0
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
    return round(entropy, 4)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance between two strings with early exit."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def extract_features(raw_url: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Extract 30 numerical, lexical, and structural features from a URL string.
    Returns a dictionary of feature names and float values, plus auxiliary metadata.
    """
    url = str(raw_url).strip()
    if not url.startswith(('http://', 'https://')):
        parsed_url_str = 'http://' + url
    else:
        parsed_url_str = url

    try:
        parsed = urlparse(parsed_url_str)
        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
    except Exception:
        hostname = url.split('/')[0] if '/' in url else url
        path = ''
        query = ''

    url_lower = url.lower()
    host_lower = hostname.lower()

    # 1-4. Length Metrics
    url_len = len(url)
    host_len = len(hostname)
    path_len = len(path)
    query_len = len(query)

    # 5-6. Structural Depth
    subdomain_parts = host_lower.split('.')
    subdomain_count = max(0, len(subdomain_parts) - 2) if len(subdomain_parts) > 2 else 0
    path_depth = path.count('/')

    # 7-10. Digit & Case Counts
    digit_count = sum(c.isdigit() for c in url)
    digit_ratio = round(digit_count / url_len, 4) if url_len > 0 else 0.0
    uppercase_count = sum(c.isupper() for c in url)
    uppercase_ratio = round(uppercase_count / url_len, 4) if url_len > 0 else 0.0

    # 11-17. Special Character Metrics
    special_char_count = sum(url.count(c) for c in SPECIAL_CHARS)
    dot_count = url.count('.')
    hyphen_count = url.count('-')
    slash_count = url.count('/')
    at_count = url.count('@')
    question_count = url.count('?')
    equal_count = url.count('=')

    # 18. Shannon Entropy
    shannon_entropy = calculate_shannon_entropy(url)

    # 19. Vowel to Consonant Ratio in Hostname
    vowels = sum(c in 'aeiou' for c in host_lower)
    consonants = sum(c in 'bcdfghjklmnpqrstvwxyz' for c in host_lower)
    vowel_consonant_ratio = round(vowels / (consonants + 1), 4)

    # 20. IP Address Host Detection
    is_ip_address = 1.0 if RE_IP.match(host_lower) else 0.0

    # 21. Punycode Detection (IDN homoglyph spoofing)
    is_punycode = 1.0 if 'xn--' in host_lower else 0.0

    # 22. Phishing Keyword Count
    suspicious_keyword_count = float(sum(1 for kw in PHISHING_KEYWORDS if kw in url_lower))

    # 23. TLD Risk Weight
    tld_risk_score = 0.1
    for tld, weight in HIGH_RISK_TLDS.items():
        if host_lower.endswith(tld):
            tld_risk_score = weight
            break

    # 24. Minimum Levenshtein Distance to Top Trusted Brands (Optimized)
    host_tokens = [t for t in RE_HOST_TOKENS.split(host_lower) if len(t) >= 3]
    
    min_brand_dist = 99
    matched_target_brand = "None"
    
    # Fast exact match check O(1)
    token_set = set(host_tokens)
    exact_matches = token_set.intersection(TOP_BRANDS_SET)
    if exact_matches:
        min_brand_dist = 0
        matched_target_brand = next(iter(exact_matches))
    else:
        for brand in TOP_BRANDS:
            b_len = len(brand)
            for token in host_tokens:
                t_len = len(token)
                if abs(t_len - b_len) >= min_brand_dist or abs(t_len - b_len) > 2:
                    continue
                dist = levenshtein_distance(token, brand)
                if dist < min_brand_dist:
                    min_brand_dist = dist
                    matched_target_brand = brand
                    if min_brand_dist <= 1:
                        break
            if min_brand_dist == 0:
                break

    is_homoglyph_brand = 1.0 if min_brand_dist in [1, 2] else 0.0

    # 25. Brand in Subdomain or Path Spoofing
    root_domain = '.'.join(subdomain_parts[-2:]) if len(subdomain_parts) >= 2 else host_lower
    has_brand_in_subdomain_or_path = 0.0
    for brand in TOP_BRANDS:
        if brand in url_lower and not root_domain.startswith(brand):
            has_brand_in_subdomain_or_path = 1.0
            break

    # 26. Authentic Brand Domain Detection
    is_authentic_brand_domain = 0.0
    if (min_brand_dist == 0 and root_domain.startswith(matched_target_brand) 
            and tld_risk_score <= 0.2 and is_ip_address == 0.0 and is_punycode == 0.0 
            and has_brand_in_subdomain_or_path == 0.0):
        is_authentic_brand_domain = 1.0

    # 27. Consecutive Repeating Characters (e.g. gggooogle)
    max_repeat = 1
    current_repeat = 1
    for i in range(1, len(url)):
        if url[i] == url[i-1]:
            current_repeat += 1
            max_repeat = max(max_repeat, current_repeat)
        else:
            current_repeat = 1
    consecutive_char_repeat_count = float(max_repeat)

    # 28. Average Word Length in URL
    words = RE_WORDS.findall(url)
    avg_word_length = round(sum(len(w) for w in words) / len(words), 4) if words else 0.0

    # 29. Character Trigram Diversity Ratio
    if len(url) >= 3:
        trigrams = [url[i:i+3] for i in range(len(url) - 2)]
        char_trigram_diversity = round(len(set(trigrams)) / len(trigrams), 4)
    else:
        char_trigram_diversity = 1.0

    feature_dict = {
        'url_length': float(url_len),
        'hostname_length': float(host_len),
        'path_length': float(path_len),
        'query_length': float(query_len),
        'subdomain_count': float(subdomain_count),
        'path_depth': float(path_depth),
        'digit_count': float(digit_count),
        'digit_ratio': float(digit_ratio),
        'uppercase_count': float(uppercase_count),
        'uppercase_ratio': float(uppercase_ratio),
        'special_char_count': float(special_char_count),
        'dot_count': float(dot_count),
        'hyphen_count': float(hyphen_count),
        'slash_count': float(slash_count),
        'at_count': float(at_count),
        'question_count': float(question_count),
        'equal_count': float(equal_count),
        'shannon_entropy': float(shannon_entropy),
        'vowel_consonant_ratio': float(vowel_consonant_ratio),
        'is_ip_address': float(is_ip_address),
        'is_punycode': float(is_punycode),
        'suspicious_keyword_count': float(suspicious_keyword_count),
        'tld_risk_score': float(tld_risk_score),
        'min_brand_levenshtein_distance': float(min_brand_dist if min_brand_dist != 99 else 10.0),
        'is_homoglyph_brand': float(is_homoglyph_brand),
        'has_brand_in_subdomain_or_path': float(has_brand_in_subdomain_or_path),
        'is_authentic_brand_domain': float(is_authentic_brand_domain),
        'consecutive_char_repeat_count': float(consecutive_char_repeat_count),
        'avg_word_length': float(avg_word_length),
        'char_trigram_diversity': float(char_trigram_diversity)
    }

    metadata = {
        'target_brand': matched_target_brand if min_brand_dist <= 2 else 'None',
        'is_homoglyph': bool(is_homoglyph_brand),
        'is_authentic_brand': bool(is_authentic_brand_domain),
        'root_domain': root_domain
    }

    return feature_dict, metadata


FEATURE_NAMES = [
    'url_length', 'hostname_length', 'path_length', 'query_length',
    'subdomain_count', 'path_depth', 'digit_count', 'digit_ratio',
    'uppercase_count', 'uppercase_ratio', 'special_char_count',
    'dot_count', 'hyphen_count', 'slash_count', 'at_count',
    'question_count', 'equal_count', 'shannon_entropy',
    'vowel_consonant_ratio', 'is_ip_address', 'is_punycode',
    'suspicious_keyword_count', 'tld_risk_score',
    'min_brand_levenshtein_distance', 'is_homoglyph_brand',
    'has_brand_in_subdomain_or_path', 'is_authentic_brand_domain',
    'consecutive_char_repeat_count', 'avg_word_length', 'char_trigram_diversity'
]


def _extract_chunk(chunk: List[str]) -> List[List[float]]:
    """Helper worker to extract feature vectors for a list of URLs."""
    results = []
    for url in chunk:
        f_dict, _ = extract_features(url)
        results.append([f_dict[name] for name in FEATURE_NAMES])
    return results


def extract_features_batch(urls: List[str], chunk_size: int = 5000, n_jobs: int = -1) -> np.ndarray:
    """
    High-speed parallel batch feature extraction across all available CPU cores.
    Returns a numpy float32 feature matrix of shape (len(urls), len(FEATURE_NAMES)).
    """
    from joblib import Parallel, delayed
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    if not urls:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
    
    if has_tqdm:
        parallel_results = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(_extract_chunk)(c) for c in tqdm(chunks, desc="  Extracting Domain Features", unit="chunk")
        )
    else:
        parallel_results = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(_extract_chunk)(c) for c in chunks
        )
        
    flat_results = [vec for sublist in parallel_results for vec in sublist]
    return np.array(flat_results, dtype=np.float32)
