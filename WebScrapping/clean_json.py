import json

input_file = 'news_dataset.json'
output_file = 'news_cleaned.json'

with open(input_file, 'r', encoding='utf-8') as file:
    nested_news_data = json.load(file)

print(
    f"Original number of news items: {sum(len(news_list) for news_list in nested_news_data)}")

unique_news = []
seen_ids = set()

for news_list in nested_news_data:
    for news in news_list:
        news_id = news.get("id")
        if news_id and news_id not in seen_ids:
            unique_news.append(news)
            seen_ids.add(news_id)

print(f"Cleaned number of news items: {len(unique_news)}")

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(unique_news, file, ensure_ascii=False, indent=4)

print(f"Cleaned news data saved to {output_file}")
