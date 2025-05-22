import os

def count_exchanges(chat_content):
    lines = [line.strip() for line in chat_content.split('\n') if line.strip()]
    exchange_count = sum(1 for line in lines if line.startswith("User:") or line.startswith("AI:"))
    return exchange_count

def main():
    chat_file_path = 'sample_chats/chat.txt'
    output_file_path = 'output/exchanges_count.txt'

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            chat_content = f.read()
    except FileNotFoundError:
        print(f"Error: Chat file not found at {chat_file_path}")
        return

    total_exchanges = count_exchanges(chat_content)
    print(f"Total exchanges: {total_exchanges}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Total exchanges: {total_exchanges}\n")
        print(f"Exchange count saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_file_path}")

if __name__ == "__main__":
    main()
