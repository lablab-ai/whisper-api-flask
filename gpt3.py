import openai



def gpt3complete(speech):
    # Completion function call engine: text-davinci-002

    Platformresponse = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Write a short summary of this text: {}".format(speech),
        temperature=0.7,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        )

    return Platformresponse.choices[0].text