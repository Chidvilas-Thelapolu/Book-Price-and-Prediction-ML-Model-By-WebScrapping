import requests
from bs4 import BeautifulSoup
import csv

# URL and headers
url = "http://books.toscrape.com/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Function to map rating classes to star counts
def extract_rating_class(rating_class):
    """Maps rating class names to numeric star ratings."""
    ratings_map = {
        "One": "1 Star",
        "Two": "2 Stars",
        "Three": "3 Stars",
        "Four": "4 Stars",
        "Five": "5 Stars"
    }
    for key in ratings_map:
        if key in rating_class:
            return ratings_map[key]
    return "No Rating"

def scrape_books(url, headers):
    """Scrapes book titles, prices, and ratings from the website."""
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return []

    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    products = soup.find_all('article', class_='product_pod')

    # Extract book data
    data = []
    for product in products:
        title_element = product.find('h3').find('a')
        price_element = product.find('p', class_='price_color')
        rating_element = product.find('p', class_='star-rating')

        if title_element and price_element and rating_element:
            title = title_element['title'] if 'title' in title_element.attrs else title_element.get_text(strip=True)
            price = price_element.get_text(strip=True)
            rating = extract_rating_class(rating_element.get("class", []))
            data.append({"Title": title, "Price": price, "Rating": rating})

    return data

def save_to_csv(data, filename):
    """Saves scraped data to a CSV file."""
    if not data:
        print("No data to save.")
        return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["Title", "Price", "Rating"])
            writer.writeheader()
            writer.writerows(data)
        print(f"Data saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving data: {e}")

def display_data(data):
    """Displays data in a tabular format in the console."""
    if not data:
        print("No data available to display.")
        return

    print(f"{'Title':<70} | {'Price':<10} | {'Rating':<10}")
    print("-" * 95)
    for book in data:
        print(f"{book['Title'][:65]:<70} | {book['Price']:<10} | {book['Rating']:<10}")

# Main execution
books_data = scrape_books(url, headers)
if books_data:
    display_data(books_data)
    save_to_csv(books_data, 'books_data_with_ratings.csv')
else:
    print("No books data extracted. Please check the website structure or network issues.")
