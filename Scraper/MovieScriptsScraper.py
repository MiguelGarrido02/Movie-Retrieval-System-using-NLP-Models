
import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import quote
import re

import time

def main():
    base_script_url = "https://imsdb.com/scripts/"
    # URL with all the movies in list format
    url = "https://imsdb.com/all-scripts.html"
    csv_file = 'test3.csv'

    # Field names
    fieldnames = ['ID', 'Title', 'Script']

    # Request page content
    page = requests.get(url)
    soup = BeautifulSoup(page.text, "html.parser")

    # Find all movie links
    all_movie_links = soup.find_all("a")
    idx = 1

    # Write data to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through each movie link
        for m in range(6, len(all_movie_links)):
            # Check if it's a relevant movie link
            if "Movie Scripts" in all_movie_links[m].get("href"):
                if (all_movie_links[m].get("title")) is None:
                    continue

                # Process title
                title = re.sub('Script', '', all_movie_links[m].get("title"))
                title = re.sub(':', '', title)
                title = title.strip()

                script_url = base_script_url + re.sub(' ', '-', title) + ".html"
                print(script_url)

                # Request page content
                page2 = requests.get(script_url)
                soup2 = BeautifulSoup(page2.text, "html.parser")

                # Find the script text
                script_text = ""
                scrtext_td = soup2.find("td", class_="scrtext")
                if scrtext_td:
                    # Extract text content from the <td class="scrtext">
                    script_text = scrtext_td.get_text(separator="\n", strip=True)
                
                # Clean script text
                script_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', script_text)  # Remove special characters
                script_text = script_text.replace("\n", " ")  # Replace newlines with space
                script_text = re.sub(r'\s+', ' ', script_text)  # Replace multiple spaces with a single space

                # Clean title
                title = title.replace(",", " ")  # Remove commas from titles for CSV
                print(title)

                # Write to CSV
                writer.writerow({
                    'ID': idx,
                    'Title': title.lower(),
                    'Script': script_text.lower()
                })

                idx += 1

    print(idx)
    start_time = time.time()

if __name__ == "__main__":
    main()
