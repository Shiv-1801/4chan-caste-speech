#Data Scraper Code provided by Ivan Kisjes

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (TimeoutException,
                                      NoSuchElementException,
                                      WebDriverException,
                                      StaleElementReferenceException)
import time, re, codecs
from bs4 import BeautifulSoup
from random import randint
import undetected_chromedriver
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Constants
MAX_RETRIES = 5
WAIT_TIME = 3
BASE_DELAY = 1
MAX_DELAY = 5
THROTTLE_DELAY = 1.0
PAGE_LOAD_TIMEOUT = 30

# Regex pattern for thread URLs
THREAD_URL_PATTERN = re.compile(r'^https://archive\.4plebs\.org/pol/thread/\d+/?$')

class FourPlebsScraper:
    def __init__(self):
        self.driver = self._init_driver()
        self.session_stats = {
            'successful_requests': 0,
            'failed_requests': 0,
            'retries': 0,
            'threads_scraped': 0
        }

    def _init_driver(self):
        """Initialize the undetected Chrome driver with options"""
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        # options.add_argument('--headless')  # Disabled for debugging
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')

        driver = undetected_chromedriver.Chrome(
            options=options,
            use_subprocess=True
        )
        driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
        return driver

    def _random_delay(self):
        """Add random delay between requests"""
        delay = BASE_DELAY + randint(0, MAX_DELAY) + THROTTLE_DELAY
        logging.debug(f"Sleeping for {delay:.2f} seconds")
        time.sleep(delay)

    def _get_with_retry(self, url: str, retries: int = MAX_RETRIES) -> bool:
        """Load a URL with retry logic"""
        for attempt in range(retries):
            try:
                logging.info(f"Loading URL (attempt {attempt + 1}): {url}")
                self.driver.get(url)
                WebDriverWait(self.driver, PAGE_LOAD_TIMEOUT).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete')
                self.session_stats['successful_requests'] += 1
                return True
            except (TimeoutException, WebDriverException) as e:
                self.session_stats['failed_requests'] += 1
                self.session_stats['retries'] += 1
                wait_time = WAIT_TIME * (attempt + 1)
                logging.warning(f"Attempt {attempt + 1} failed for {url}. Error: {str(e)}")
                logging.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if attempt == retries - 1:
                    logging.error(f"Failed to load {url} after {retries} attempts")
                    return False
        return False

    def _save_page_for_debugging(self, filename: str = 'debug_page.html'):
        """Save current page source for debugging"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.driver.page_source)
        logging.info(f"Saved page source to {filename}")

    def get_thread_data(self, link: str) -> Optional[Dict]:
        """Get thread data with comprehensive error handling"""
        for attempt in range(MAX_RETRIES):
            try:
                if not self._get_with_retry(link):
                    return None

                # Wait for essential elements to load
                try:
                    WebDriverWait(self.driver, PAGE_LOAD_TIMEOUT).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'aside.posts article'))
                    )
                except TimeoutException:
                    self._save_page_for_debugging()
                    logging.warning(f"No articles found in thread: {link}")
                    return None

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                aside = soup.find('aside', {'class': 'posts'})

                if not aside:
                    logging.warning(f"No aside.posts found in thread: {link}")
                    self._save_page_for_debugging()
                    return None

                article = aside.find('article')
                if not article:
                    logging.warning(f"No article found in aside: {link}")
                    return None

                # Extract post ID
                post_id = article.get('id', '').strip()
                if not post_id:
                    logging.warning(f"No ID found for post: {link}")
                    return None

                # Extract post metadata
                post_meta = article.find('div', {'class': 'post_data'})
                if not post_meta:
                    logging.warning(f"No post_data found for post {post_id}")
                    return None

                # Extract text content
                text_div = article.find('div', {'class': 'text'})
                text = text_div.get_text(separator='\n').strip() if text_div else None

                # Extract backlinks
                backlinks = []
                backlink_div = article.find('div', {'class': 'backlink_list'})
                if backlink_div:
                    backlinks = [
                        a.get('href').split('#')[0]
                        for a in backlink_div.find_all('a', href=True)
                        if a.get('href') != '#'
                    ]

                # Extract title, author, and time
                title = None
                title_element = post_meta.find('h2', {'class': 'post_title'})
                if title_element:
                    title = title_element.get_text(strip=True)

                author = None
                author_element = post_meta.find('span', {'class': 'post_author'})
                if author_element:
                    author = author_element.get_text(strip=True)

                post_time = None
                time_element = post_meta.find('time')
                if time_element:
                    post_time = time_element.get('datetime')

                # Extract image data
                img = article.find('img')
                img_src = img.get('src') if img else None
                img_title = img.get('title') if img else None

                result = {
                    'id': post_id,
                    'text': text,
                    'backlinks': backlinks,
                    'title': title,
                    'author': author,
                    'time': post_time,
                    'img_title': img_title,
                    'img_src': img_src,
                    'url': link,
                    'scrape_time': datetime.now().isoformat()
                }

                self.session_stats['threads_scraped'] += 1
                logging.info(f"Successfully scraped thread {post_id}")
                return result

            except Exception as e:
                logging.error(f"Error processing thread {link} (attempt {attempt + 1}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    logging.error(f"Failed to process thread after {MAX_RETRIES} attempts: {link}")
                    return None
                time.sleep(WAIT_TIME * (attempt + 1))

    def has_next_page(self, soup: BeautifulSoup) -> bool:
        """Check if there's a next page available"""
        try:
            paginate_div = soup.find('div', {'class': 'paginate'})
            if not paginate_div:
                logging.debug("No paginate div found")
                return False

            last_li = paginate_div.find('ul').find_all('li')[-1]
            next_button = last_li.find('a')

            if next_button and next_button.get('href') != '#':
                logging.debug("Next page available")
                return True

            logging.debug("No next page available")
            return False
        except Exception as e:
            logging.warning(f"Error checking for next page: {str(e)}")
            return False

    def get_thread_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract thread links from search results page"""
        links = []
        try:
            aside = soup.find('aside', {'class': 'posts'})
            if not aside:
                logging.warning("No aside.posts found in search results")
                return links

            for article in aside.find_all('article'):
                post_data = article.find('div', {'class': 'post_data'})
                if not post_data:
                    continue

                for a in post_data.find_all('a', href=True):
                    href = a['href']
                    if href and href != '#':
                        # Clean and validate URL
                        clean_url = href.split('#')[0]
                        if not clean_url.endswith('/'):
                            clean_url += '/'

                        if THREAD_URL_PATTERN.match(clean_url) and clean_url not in links:
                            links.append(clean_url)
                            logging.debug(f"Found thread link: {clean_url}")

            logging.info(f"Found {len(links)} thread links on this page")
            return list(set(links))  # Remove duplicates

        except Exception as e:
            logging.error(f"Error extracting links: {str(e)}")
            return links

    def scrape_search_results(self, search_term: str, year: int, month: int) -> List[str]:
        """Scrape all thread links for a given search term and month"""
        month_str = str(month).zfill(2)
        end_date = (datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1))

        base_url = (
            f"https://archive.4plebs.org/pol/search/text/{search_term}/"
            f"start/{year}-{month_str}-01/end/{end_date.date()}/"
        )

        all_links = []
        page = 1

        while True:
            url = base_url if page == 1 else f"{base_url}page/{page}/"
            logging.info(f"Processing page {page}: {url}")

            if not self._get_with_retry(url):
                break

            try:
                # Wait for either articles or "no results" message
                WebDriverWait(self.driver, PAGE_LOAD_TIMEOUT).until(
                    lambda d: d.find_elements(By.CSS_SELECTOR, "aside.posts article") or
                            d.find_elements(By.CSS_SELECTOR, "div.alert")
                )

            except TimeoutException:
                logging.warning(f"Timeout waiting for content on {url}")
                self._save_page_for_debugging(f"timeout_page_{page}.html")
                break

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            # Check for no results
            alert = soup.find('div', {'class': 'alert'})
            if alert and 'No results found' in alert.text:
                logging.info('No results found for this time period')
                break

            # Get links from current page
            page_links = self.get_thread_links(soup)
            all_links.extend(page_links)
            all_links = list(set(all_links))  # Deduplicate

            logging.info(f"Page {page}: Found {len(page_links)} new links (Total: {len(all_links)})")

            # Check for next page
            if not self.has_next_page(soup):
                break

            page += 1
            self._random_delay()

        return all_links

    def scrape_search_term(self, search_term: str, start_year: int, end_year: int):
        """Main scraping function for a search term"""
        all_threads = []

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == end_year and month > datetime.now().month:
                    break

                logging.info(f"Processing {search_term} - {year}-{month:02d}")
                try:
                    month_links = self.scrape_search_results(search_term, year, month)
                    logging.info(f"Found {len(month_links)} threads for {year}-{month:02d}")

                    # Process each thread
                    for i, link in enumerate(month_links, 1):
                        logging.info(f"Processing thread {i}/{len(month_links)}: {link}")
                        thread_data = self.get_thread_data(link)
                        if thread_data:
                            all_threads.append(thread_data)
                        self._random_delay()

                except Exception as e:
                    logging.error(f"Error processing {year}-{month:02d}: {str(e)}")
                    continue

        # Save results
        if all_threads:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{search_term}_results_{timestamp}.csv"

            # Ensure all records have the same fields
            fieldnames = set()
            for record in all_threads:
                fieldnames.update(record.keys())
            fieldnames = sorted(fieldnames)

            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_threads)

            logging.info(f"Saved {len(all_threads)} records to {filename}")
        else:
            logging.warning(f"No data collected for search term: {search_term}")

        return all_threads

    def close(self):
        """Clean up resources"""
        try:
            self.driver.quit()
            logging.info("WebDriver closed successfully")
        except Exception as e:
            logging.error(f"Error closing driver: {str(e)}")

def main():
    searches = ['caste']  # Add more terms as needed
    start_year = 2013
    end_year = 2024

    scraper = FourPlebsScraper()

    try:
        for search_term in searches:
            logging.info(f"Starting scrape for: {search_term}")
            results = scraper.scrape_search_term(search_term, start_year, end_year)
            logging.info(f"Found {len(results)} threads for {search_term}")

        logging.info("Scraping complete!")
        logging.info(f"Session statistics: {scraper.session_stats}")

    except KeyboardInterrupt:
        logging.info("Scraping interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main()

#Cleaning and pre-processing the data 

!python -m spacy download en_core_web_lg
import codecs, csv, re, pickle, sys
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from operator import itemgetter
from string import punctuation
from gensim.models import Phrases
import spacy

nlp = spacy.load('en_core_web_lg')

bigram=Phrases.load('Downloads/phrases.bin')

eng_stops = stopwords.words('english')
eng_stops.append('www')
eng_stops += ["'s", "n't", "https", "'re", '...']
eng_stops += [a for a in punctuation]
eng_stops= set(eng_stops)

word_dict={}

def clean(value):
    global eng_stops, word_dict, nlp
    value=value.lower()
    value=re.sub(r'"""|\\|"|>|<|▶|■|™|□|_|-|1|2|3|4|5|6|7|8|9|0', '', value)
    # print(value)
    doc = nlp(value)
    words=[]
    for token in doc:
        if not token.lemma_ in eng_stops:
            if token.pos_ in ['VERB', 'PROPN', 'NOUN']:
                words.append(token.lemma_)
    for w in set(words):
        try:
            word_dict[w]+=1
        except KeyError:
            word_dict[w]=1
    # print(words)
    # print('---------------------------------------')
    return words

data=[]
word_data=[]
ct=0
with codecs.open("# insert file path",'r', encoding='utf-8') as f:
    r=csv.DictReader(f)
    for row in r:
        ct+=1
        if ct % 10000 == True:
            print('-', ct/1000)
        res = clean(row['text'])
        word_data.append(bigram[res])
        # if ct > 5000:
            # break

if False:#to create phrases
    bigram = Phrases(word_data, min_count=2, threshold=2)
    bigram.save('Downloads/phrases.bin')

total_posts = len(word_data)

word_data = pd.DataFrame(word_data)
word_data.to_csv('insert file path')

#Custom Lexicon implementation and Hate Speech Detection by Claude Sonnet 4

class CasteHateSpeechDetector:
    """
    A class for detecting caste mentions and hate speech in text data.
    This is a simplified version focusing only on steps 2 and 3 of the pipeline.
    """

    def __init__(self, data_path):
        """
        Initialize the detector with data path.

        Args:
            data_path (str): Path to the preprocessed 4chan /pol/ data
        """
        self.data_path = data_path
        self.data = None

        # Define caste-related terms dictionary for India
        self.caste_terms = {
            'upper_caste': [
                'brahmin', 'brahmins', 'kshatriya', 'kshatriyas', 'vaishya', 'vaishyas',
                'dvija', 'twice-born', 'savarna', 'savarnas', 'forward caste',
                'upper caste', 'general category', 'upper class', 'higher caste',
                'brahmin community', 'brahmanical', 'brahminical', 'bhumihar', 'rajput','brahman'
            ],
            'lower_caste': [
                'dalit', 'dalits', 'shudra', 'shudras', 'untouchable', 'untouchables',
                'sc', 'scheduled caste', 'scheduled castes', 'st', 'scheduled tribe',
                'scheduled tribes', 'obc', 'other backward class', 'backward caste',
                'backward class', 'lower caste', 'harijan', 'harijans', 'depressed class',
                'valmiki', 'chamar', 'dhobi', 'bhangi', 'mahar', 'mala', 'madiga', 'pasi',
                'paraiyar', 'pallar', 'adivasi', 'tribal', 'bahujan', 'mahadalit'
            ],
            'caste_system': [
                'caste', 'castes', 'caste system', 'jati', 'varna', 'chaturvarna', 'varnashrama', 'endogamy',
                'inter-caste', 'intercaste', 'casteism', 'caste discrimination', 'caste violence',
                'caste hierarchy', 'hereditary occupation', 'reservation', 'quota',
                'affirmative action', 'manual scavenging', 'caste privilege',
                'caste oppression', 'caste atrocity', 'caste identity', 'caste census'
            ],
            'slurs': [
                 'jeet', 'pajeet', 'mudslime'
            ]
        }

        # Combine all caste terms for general detection
        self.all_caste_terms = []
        for category in self.caste_terms.values():
            self.all_caste_terms.extend(category)

        # Common hate speech indicators
        self.hate_indicators = [
            'kill', 'die', 'hate', 'filthy', 'dirty', 'disgusting', 'inferior',
            'subhuman', 'destroy', 'eliminate', 'get rid of', 'vermin', 'parasite',
            'disease', 'exterminate', 'genocide', 'worthless', 'scum', 'trash',
            'cleanse', 'purge', 'unclean', 'pollute', 'contaminate', 'impure',
            'swine', 'animal', 'beast', 'savage', 'primitive', 'backward'
        ]

    def load_data(self):
        """
        Load data from file.
        """
        print("Loading data...")

        # Determine file format from extension
        file_extension = self.data_path.split('.')[-1].lower()

        if file_extension == 'csv':
            self.data = pd.read_csv(self.data_path)
        elif file_extension in ['json', 'jsonl']:
            self.data = pd.read_json(self.data_path, lines=(file_extension == 'jsonl'))
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        # Standardize text column name
        if 'text' not in self.data.columns:
            if 'content' in self.data.columns:
                self.data['text'] = self.data['content']
            elif 'body' in self.data.columns:
                self.data['text'] = self.data['body']

        # Ensure text is string type
        self.data['text'] = self.data['text'].astype(str)

        # Clean text for analysis
        self.data['cleaned_text'] = self.data['text'].apply(self.clean_text)

        print(f"Data loaded. Total posts: {len(self.data)}")
        return self.data

    def clean_text(self, text):
        """
        Basic text cleaning.

        Args:
            text (str): Input text

        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def detect_caste_mentions(self):
        """
        STEP 2: Detect mentions of caste-related terms in the text.
        """
        print("Detecting caste mentions...")

        # Function to check for term presence
        def check_terms(text, term_list):
            text = text.lower()
            # Check for exact matches with word boundaries
            for term in term_list:
                if re.search(r'\b' + re.escape(term) + r'\b', text):
                    return True
            return False

        # Check for each category of caste terms
        for category, terms in self.caste_terms.items():
            column_name = f'has_{category}_mention'
            self.data[column_name] = self.data['cleaned_text'].apply(lambda x: check_terms(x, terms))

        # Create a general caste mention flag
        self.data['has_caste_mention'] = self.data['cleaned_text'].apply(lambda x: check_terms(x, self.all_caste_terms))

        # Count mentions across categories
        caste_mentions = {
            category: self.data[f'has_{category}_mention'].sum()
            for category in self.caste_terms.keys()
        }

        print("\nCaste mentions detected:")
        for category, count in caste_mentions.items():
            print(f"- {category}: {count} posts")

        total_caste_posts = self.data['has_caste_mention'].sum()
        print(f"\nTotal posts with caste mentions: {total_caste_posts} ({total_caste_posts/len(self.data)*100:.2f}% of dataset)")

        # Find most frequently co-occurring terms with caste mentions
        # This helps identify context around caste discussions
        self.analyze_co_occurring_terms()

        return self.data


    def detect_hate_speech(self):
        """
        STEP 3: Detect potential hate speech based on hate indicators near caste terms.
        """
        print("\nDetecting potential hate speech...")

        # Function to check for hate indicators near caste terms
        def check_hate_proximity(text, caste_terms, hate_terms, window_size=10):
            text = text.lower()
            words = text.split()

            # Check if any caste term is present
            caste_indices = []
            for i, word in enumerate(words):
                for term in caste_terms:
                    # Handle multi-word terms
                    if ' ' in term:
                        # Check if this position could start a multi-word match
                        if i <= len(words) - len(term.split()):
                            multi_word = ' '.join(words[i:i+len(term.split())])
                            if term in multi_word:
                                caste_indices.append(i)
                    # Single-word term
                    elif re.search(r'\b' + re.escape(term) + r'\b', word):
                        caste_indices.append(i)

            # If no caste terms, return False
            if not caste_indices:
                return False

            # Check for hate indicators within window_size words of caste terms
            for caste_idx in caste_indices:
                start = max(0, caste_idx - window_size)
                end = min(len(words), caste_idx + window_size + 1)
                window = words[start:end]

                for term in hate_terms:
                    # Handle multi-word hate terms
                    if ' ' in term:
                        window_text = ' '.join(window)
                        if term in window_text:
                            return True
                    # Single-word term
                    elif any(re.search(r'\b' + re.escape(term) + r'\b', word) for word in window):
                        return True

            return False

        # Check for hate speech indicators near caste terms
        self.data['potential_hate_speech'] = self.data['cleaned_text'].apply(
            lambda x: check_hate_proximity(x, self.all_caste_terms, self.hate_indicators)
        )

        # Create more specific hate speech detection for each caste category
        for category, terms in self.caste_terms.items():
            column_name = f'hate_speech_against_{category}'
            self.data[column_name] = self.data['cleaned_text'].apply(
                lambda x: check_hate_proximity(x, terms, self.hate_indicators)
            )

        # Count potential hate speech instances
        hate_speech_count = self.data['potential_hate_speech'].sum()
        print(f"Potential hate speech detected in {hate_speech_count} posts ({hate_speech_count/len(self.data)*100:.2f}% of dataset)")

        # Count by category
        for category in self.caste_terms.keys():
            count = self.data[f'hate_speech_against_{category}'].sum()
            posts_with_category = self.data[f'has_{category}_mention'].sum()
            percentage = (count / posts_with_category * 100) if posts_with_category > 0 else 0
            print(f"- Against {category}: {count} posts ({percentage:.2f}% of mentions)")

        # Identify most common hate indicators in caste-related contexts
        self.analyze_hate_indicators()

        return self.data

    def analyze(self):
        """
        Run the full detection pipeline.
        """
        self.load_data()
        self.detect_caste_mentions()
        self.detect_hate_speech()

        # Get distribution of hate speech by caste category
        caste_categories = list(self.caste_terms.keys())
        total_hate = self.data['potential_hate_speech'].sum()

        print("\nDistribution of hate speech by caste category:")
        for category in caste_categories:
            count = self.data[f'hate_speech_against_{category}'].sum()
            percentage = (count / total_hate * 100) if total_hate > 0 else 0
            print(f"- {category}: {count} posts ({percentage:.2f}% of all hate speech)")

if __name__ == "__main__":
    # Replace with your actual data path
    detector = CasteHateSpeechDetector("insert file path")
    detector.analyze()

    # Export results if needed
    detector.data.to_csv("insert file path", index=False)

    print("\nAnalysis complete. Results saved to 'insert file path'")

#Visualizing results

# Identify boolean columns related to hate speech detection
boolean_cols = [col for col in results_df.columns if col.startswith('has_') or col.startswith('hate_speech_against_') or col == 'potential_hate_speech']

# Calculate the sum of True values for each boolean column
boolean_sums = results_df[boolean_cols].sum()

# Calculate the total number of posts
total_posts = len(results_df)

# Calculate the percentage for each boolean column
boolean_percentages = (boolean_sums / total_posts) * 100

# Sort the percentages for better visualization
boolean_percentages = boolean_percentages.sort_values(ascending=False)

# Generate the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x=boolean_percentages.index, y=boolean_percentages.values, palette='viridis')
plt.xlabel("Category")
plt.ylabel("Percentage of Total Posts (%)")
plt.title("Percentage of Total Posts by Hate Speech/Caste Mention Category")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Downloads/Percentage_bar_graph.png')
plt.show()

#Modify this section to get results for specific categories

#TF-IDF process
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pickle

vectorizer = TfidfVectorizer( ngram_range=(1,2), max_features=500000)
tfidf_matrix = vectorizer.fit_transform(data)
xtfidf_norm = normalize(tfidf_matrix, norm='l1', axis=1)

n_components = 10  # You can experiment with different values
nmf_model = NMF(n_components=n_components, random_state=42,  verbose=1)
nmf_matrix = nmf_model.fit(xtfidf_norm)

#save the model
pickle.dump(nmf_matrix, open('Downloads/nmf_matrix_phr.bin','wb'))

# Visualize TF-IDF as a bar graph of top 10 words
feature_names = vectorizer.get_feature_names_out()
# Calculate sum of scores directly on the sparse matrix
tfidf_scores = tfidf_matrix.sum(axis=0).A1 # .A1 converts the result to a 1D numpy array
top_10_indices = tfidf_scores.argsort()[-10:][::-1]
top_10_words = [feature_names[i] for i in top_10_indices]
top_10_scores = [tfidf_scores[i] for i in top_10_indices]

plt.figure(figsize=(10, 6))
plt.bar(top_10_words, top_10_scores)
plt.xlabel('Words')
plt.ylabel('TF-IDF Score')
plt.title('Top 10 TF-IDF Words')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Parameter Selection for NMF (using coherence score - requires gensim)

from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_values(tfidf, dictionary, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = NMF(n_components=num_topics, random_state=42)
        model.fit(tfidf)
        model_list.append(model)
        coherencemodel = CoherenceModel(topics=get_topics(model, vectorizer), texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

def get_topics(model, vectorizer):
    topics = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -10 -1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        topics.append(top_features)
    return topics


# Tokenize texts for coherence score
texts = [[word for word in document.split()] for document in data['text']]
from gensim.corpora.dictionary import Dictionary
id2word = Dictionary(texts)

model_list, coherence_values = compute_coherence_values(tfidf=tfidf, dictionary=id2word, texts=texts, start=2, limit=20, step=1)

# Find optimal number of topics
limit=20; start=2; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Apply NMF with the optimal number of topics (e.g., the one with highest coherence)
optimal_n_topics = np.argmax(coherence_values) + 2 # Add 2 because start=2
optimal_model = model_list[optimal_n_topics - 2] # Adjust indexing since start=2

# Get the topics
topics = get_topics(optimal_model, vectorizer)
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {', '.join(topic)}")

#NMF Topic Modelling

def get_nmf_topics(model, n_top_words):
    global n_components, vectorizer
    #the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
    feat_names = vectorizer.get_feature_names_out()

    word_dict = {};
    for i in range(n_components):

        #for each topic, obtain the largest values, and add the words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;

    return pd.DataFrame(word_dict)

topics = get_nmf_topics(nmf_matrix, 20)

#create spreadsheet
topics.to_csv('insert file name')

# Visualize NMF topics as word clouds
!pip install wordcloud
from wordcloud import WordCloud

def show_wordcloud(topic_words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(topic_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

for i in range(n_components):
    print(f"Topic #{i+1}")
    # Access the topic column using the zero-padded format
    topic_words = topics[f'Topic # {i+1:02d}'].tolist()
    show_wordcloud(topic_words)
