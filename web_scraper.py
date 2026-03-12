import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def scrape_catalog():
    base_url = "https://www.shl.com"
    # type=1 filters for "Individual Test Solutions"
    catalog_url = "https://www.shl.com/products/product-catalog/?start={}&type=1"
    
    # HEADERS to bypass basic bot protection
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }
    
    all_assessments = []
    start = 0
    
    print("Scraping main catalog table...")
    
    while True:
        print(f"Fetching rows starting at {start}...")
        response = requests.get(catalog_url.format(start), headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Target only the rows with data-entity-id (Individual Test Solutions)
        rows = soup.find_all('tr', attrs={'data-entity-id': True})
        
        if not rows:
            print("No more rows found (or blocked). Exiting pagination loop.")
            break 
            
        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 4:
                continue
                
            # Extract Name and URL from the first cell
            a_tag = cells[0].find('a')
            if not a_tag:
                continue
                
            name = a_tag.text.strip()
            link = a_tag['href']
            full_url = f"{base_url}{link}" if link.startswith('/') else link
            
            # Extract Yes/No for Remote and Adaptive Support
            remote_support = "Yes" if cells[1].find('span', class_='-yes') else "No"
            adaptive_support = "Yes" if cells[2].find('span', class_='-yes') else "No"
            
            # Extract Test Types (e.g., ['A', 'E', 'B', 'P'])
            test_types = [span.text.strip() for span in cells[3].find_all('span', class_='product-catalogue__key')]
            
            all_assessments.append({
                "name": name,
                "url": full_url,
                "remote_support": remote_support,
                "adaptive_support": adaptive_support,
                "test_type": test_types
            })
            
        # Check if there is a "Next" button and if it is active
        next_button = soup.find('li', class_='pagination__item -arrow -next')
        if not next_button or 'disabled' in next_button.get('class', []):
            break
            
        start += 12
        time.sleep(1)
        
    print(f"Successfully extracted {len(all_assessments)} assessments.")
    
    if len(all_assessments) == 0:
        print("ERROR: SHL is still blocking the request. We may need to use Selenium instead.")
        return

    print("\nFetching descriptions and durations from individual pages...")
    for i, assessment in enumerate(all_assessments):
        try:
            res = requests.get(assessment['url'], headers=headers)
            detail_soup = BeautifulSoup(res.content, 'html.parser')
            
            # Extract Description
            desc_heading = detail_soup.find(lambda tag: tag.name == "h4" and "Description" in tag.text)
            if desc_heading and desc_heading.find_next_sibling('p'):
                assessment['description'] = desc_heading.find_next_sibling('p').text.strip()
            else:
                assessment['description'] = "Description not available."
                
            # Extract Duration
            duration_heading = detail_soup.find(lambda tag: tag.name == "h4" and "Assessment length" in tag.text)
            if duration_heading and duration_heading.find_next_sibling('p'):
                duration_text = duration_heading.find_next_sibling('p').text
                nums = re.findall(r'\d+', duration_text)
                assessment['duration'] = int(nums[0]) if nums else 0
            else:
                assessment['duration'] = 0
                
        except Exception as e:
            assessment['description'] = "Error fetching description."
            assessment['duration'] = 0
            
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(all_assessments)} individual pages...")
            
    # Save to CSV
    df = pd.DataFrame(all_assessments)
    df.to_csv("shl_assessments.csv", index=False)
    print(f"\nSuccess! Data saved to shl_assessments.csv. Total rows: {len(df)}")

if __name__ == "__main__":
    scrape_catalog()