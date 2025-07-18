from crawlers.Crawler import Crawler
import requests
import bs4
import uuid


class ElDiario(Crawler):
    def __init__(self, url):
        super().__init__(url)
        self.newspaper = "EL DIARIO"

    def crawl(self):
        def get_article_body(url):
            try: 
                resp = requests.get(url)
                soup = bs4.BeautifulSoup(resp.text, "html.parser")
                # footer
                footer_ul = soup.find("ul", class_="footer")
                if footer_ul: 
                    body = footer_ul.find("h2").get_text()
                else :
                    body = ""

                # texto 
                parragraphs = soup.find_all("p", class_="article-text")
                for p in parragraphs:
                    body += "\n"
                    body += p.get_text()
            except:
                body = "N/A"


            return body

        data = []
        response = requests.get(self.url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Cover stories
        main_divs = soup.find_all("div", class_="md__new")
        for main_div in main_divs:
            h2 = main_div.find(
                "h2", class_="ni-title")
            if not h2:
                print("No h2")
                continue

            a_tag = h2.find("a")

            link = a_tag["href"]
            # en ElDiario, hay algunas noticias en las que el link no es completo, hay que agregarle al principio el dominio
            if link.startswith("/"):
                link = self.url + link
            headline = a_tag.get_text(strip=True)
            
            body = get_article_body(link)
            if body == "N/A" or body == "":
                # descartamos noticias sin cuerpo
                continue

            unique_id = str(uuid.uuid4())
            data.append({"id": unique_id, "headline": headline, "body": body,
                        "link": link,"fecha": self.fecha, "sesgo": "N" ,"newspaper": self.newspaper})
            
        return data