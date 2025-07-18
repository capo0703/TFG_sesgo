from crawlers.Crawler import Crawler
import requests
import bs4
import uuid


class ElConfidencial(Crawler):
    def __init__(self, url):
        super().__init__(url)
        self.newspaper = "EL CONFIDENCIAL"

    def crawl(self):
        def get_article_body(url):
            try:
                resp = requests.get(url)
                soup = bs4.BeautifulSoup(resp.text, "html.parser")
                body = ""

                #body 
                body_div = soup.find("div", class_="news-body-complete")
                if not body_div:
                    return "N/A"
              
                # texto
                parragraphs = body_div.find_all("p")
                j = 0
                for p in parragraphs:
                    j += 1
                  

                    body += "\n"
                    body += p.get_text()
            except Exception as e:
                body = "N/A"

            return body

        data = []
        response = requests.get(self.url)
        
      
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # get articles
        articles = soup.find_all("article")
        
        i = 0
        for article in articles:
            a_tag = article.find("a")
            if not a_tag:
                continue

            link = a_tag["href"]

            if link.startswith("/"):
                link = self.url + link

            # revisar si el dominio es el confidencial, si no, pasar al siguiente (publicidad)

            if "https://www.elconfidencial.com" not in link:
                continue
            
            headline = a_tag.get_text(strip=True)
 
            body = get_article_body(link)
            if body == "N/A" or body == "":
                continue
            
            i += 1
            
            unique_id = str(uuid.uuid4())
            data.append({"id": unique_id, "headline": headline, "body": body,
                        "link": link,"fecha": self.fecha, "sesgo": "N" ,"newspaper": self.newspaper})
            
        return data