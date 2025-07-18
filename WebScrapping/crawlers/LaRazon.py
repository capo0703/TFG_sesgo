from crawlers.Crawler import Crawler
import requests
import bs4
import uuid


class LaRazon(Crawler):
    def __init__(self, url):
        super().__init__(url)
        self.newspaper = "LA RAZON"

    def crawl(self):
        def get_article_body(url):
            try:
                resp = requests.get(url)
                soup = bs4.BeautifulSoup(resp.text, "html.parser")
                body = ""

                #body 
                body_div = soup.find("div", class_="article-main__content")
                # texto
                parragraphs = body_div.find_all("p")

                for p in parragraphs:
                    body += "\n"
                    body += p.get_text()
            except:
                body = "N/A"

            return body

        data = []
        response = requests.get(self.url)
      
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # get articles
        articles = soup.find_all("h2", class_="article__title")
        
        for article in articles:
  
            a_tag = article.find("a")

            link = a_tag["href"]

            if link.startswith("/"):
                link = self.url + link
            
            headline = a_tag.get_text(strip=True)

            body = get_article_body(link)
            if body == "N/A" or body == "":
                continue

            unique_id = str(uuid.uuid4())
            data.append({"id": unique_id, "headline": headline, "body": body,
                        "link": link,"fecha": self.fecha, "sesgo": "N" ,"newspaper": self.newspaper})
            
        return data