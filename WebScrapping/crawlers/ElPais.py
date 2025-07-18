from crawlers.Crawler import Crawler
import requests
import bs4
import uuid


class ElPais(Crawler):
    def __init__(self, url):
        super().__init__(url)
        self.newspaper = "EL PAIS"

    def crawl(self):
        def get_article_body(url):
            try:
                resp = requests.get(url)
                soup = bs4.BeautifulSoup(resp.text, "html.parser")
                body = ""

                # abstract 
                abstract = soup.find("h2", class_="a_st")  
                if abstract:
                    body += abstract.get_text()
                body += "\n"

                # texto
                parragraphs = soup.find_all("p")
                for p in parragraphs:
                    body += p.get_text()
                    body += "\n"
            except:
                body = "N/A"

            return body

        data = []
        response = requests.get(self.url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # get articles
        articles = soup.find_all("article")

        for article in articles:
            h2 = article.find("h2", class_="c_t")

            if not h2:
                continue

            a_tag = h2.find("a")

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