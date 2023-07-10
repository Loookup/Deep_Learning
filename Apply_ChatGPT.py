import openai

openai.api_key = "Your API Key"

message_list = []
while True:

    Input = input("->")

    if Input == "":
        break

    message_list.append(Input)

print(message_list)

message = message_list[0]


completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
     
            "role": "system",
            "content": "You are a kind elementary school teacher talking to your students "
 
        },
        {
    
            "role": "user",
            "content": message

        },
    ],
)
print(completion)
