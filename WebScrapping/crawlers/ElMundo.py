from crawlers.Crawler import Crawler
import requests
import bs4
import uuid


class ElMundo(Crawler):
    def __init__(self, url):
        super().__init__(url)
        self.newspaper = "EL MUNDO"

    def crawl(self):
        def get_article_body(url):
            try:
                resp = requests.get(url)
                detailsoup = bs4.BeautifulSoup(resp.text, "html.parser")
                # Might be multiple <div> with class="ue-c-article__body" data-section="articleBody"
                body_divs = detailsoup.find_all(
                    "div", class_="ue-c-article__body", attrs={"data-section": "articleBody"})
                if not body_divs:
                    return "N/A"
                all_paragraphs = []
                for div in body_divs:
                    for p in div.find_all("p"):
                        all_paragraphs.append(p.get_text(strip=True))
                return " ".join(all_paragraphs) if all_paragraphs else "N/A"
            except:
                return "N/A"

        data = []
        response = requests.get(self.url)
        soup = bs4.BeautifulSoup(response.text, "html.parser")

        # Cover stories
        main_divs = soup.find_all("div", class_="ue-c-cover-content__main")
        for main_div in main_divs:
            a_tag = main_div.find(
                "a", class_="ue-c-cover-content__link", href=True)
            if not a_tag:
                continue

            link = a_tag["href"]
            if link.startswith("/"):
                link = self.url + link

            h2 = a_tag.find("h2", class_="ue-c-cover-content__headline")
            if not h2:
                continue
            headline = h2.get_text(strip=True)
           
            body = get_article_body(link)
            if body == "N/A" or body == "":
                # descartamos noticias sin cuerpo
                continue
            unique_id = str(uuid.uuid4())
            data.append({"id": unique_id, "headline": headline, "body": body,
                        "link": link,"fecha": self.fecha, "sesgo": "N" ,"newspaper": self.newspaper})

        # Related news
        footer_divs = soup.find_all("div", class_="ue-c-cover-content__footer")
        for footer_div in footer_divs:
            related_items = footer_div.find_all(
                "li", class_="ue-c-cover-content__related-link")
            for li in related_items:
                a_tag = li.find("a", href=True)
                if not a_tag:
                    continue
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