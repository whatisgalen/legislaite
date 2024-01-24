import google.generativeai as genai
import os

os.environ['API_KEY'] = ""
genai.configure(api_key=os.environ['API_KEY'])

response = genai.chat(messages=[
"""
What law in the US supreme court
"""
])
# print(response)
# print(response.last) #  'Hello! What can I help you with?'
# Open the file in write mode
file = open("/Users/galenmancino/repos/legislaite/chatoutput.txt", "w")

# Write the string to the file
string = response.last
file.write(string)

# Close the file
file.close()