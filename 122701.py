import requests
from bs4 import BeautifulSoup
def write(file_path, text_to_append):
    try:
        with open(file_path, 'a', encoding='utf-8') as file:
            file.write(text_to_append)
            #print("Text appended successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
def clean(html_content):
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_content, 'html.parser')

    # Get plain text from the HTML content
    plain_text = soup.get_text(separator=' ', strip=True)

    return plain_text

def main():
    for i in range(1,42):
        print("Downloading: ",i,"/41")
        response=requests.get("https://www.govinfo.gov/content/pkg/USCODE-2022-title{}/html/USCODE-2022-title{}.htm".format(i,i))
        write("usc.txt",clean(response.content)+'\n')
if  __name__ == "__main__":
    main()