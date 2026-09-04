from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup


def extract_html_from_mhtml(file_path: str) -> str:
    with open(file_path, "rb") as f:
        message = BytesParser(policy=policy.default).parse(f)

    for part in message.walk():
        if part.get_content_type() == "text/html":
            return part.get_payload(decode=True).decode("utf-8", errors="replace")

    raise ValueError("No HTML part found")


def extract_article_text(file_path: str) -> str:
    html = extract_html_from_mhtml(file_path)

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that don't contain useful article text
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Get the main article only
    article = soup.find("div", id="article")

    if article is None:
        raise ValueError("Could not find article content")

    # Extract readable text
    text = article.get_text(separator="\n", strip=True)

    return text

def extract_article_html(file_path: str) -> str:
    html = extract_html_from_mhtml(file_path)

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that don't contain useful article content
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Get the main article
    article = soup.find("div", id="article")

    if article is None:
        raise ValueError("Could not find article content")

    # Remove Investing.com promotional widgets
    for widget in article.find_all(
        attrs={"data-article-widget": "contextual-hook"}
    ):
        widget.decompose()

    # # Remove Investing.com interactive stock widget
    # text = article.find(
    #     string=lambda s: s and "Included in our AI-picked strategies" in s
    # )

    # if text:
    #     widget = text.parent.parent.parent
    #     widget.decompose()

    # Normalize headings so their text is directly inside h1/h2/h3
    for heading in article.find_all(["h1", "h2", "h3"]):
        heading_text = heading.get_text(" ", strip=True)
        heading.clear()
        heading.append(heading_text)    

    return str(article)