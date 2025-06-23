from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HF_TOKEN")

# this is a generic LLM wrapper, for text generation.

llm = HuggingFaceEndpoint(        
    repo_id = "microsoft/Phi-3-mini-4k-instruct",
    task ="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat = ChatHuggingFace(llm=llm, verbose=True) # this wraps the LLM for chat-like input, i.e., multi-turn messages


messages = [
    # ("system", "You are a helpful translator. Translate the user sentence to French."),
    # ("human", "I love programming."),
    # ("human", "my name is rudra"),
    SystemMessage(content = "You are an expert in social media marketing content strategy"),
    HumanMessage(content = "Give a short tip on engaging Instagram posts. Also, give ad/reel ideas to promote small businesses or shops."), 
    AIMessage(content = '''Tip for Engaging Instagram Posts:
- Use Dynamically Captioned Stories: Briefly captivate your audience by posting interesting questions or fast facts directly in your stories, promoting engagement through replies and responses.

Ad/Reel Ideas for Promoting Small Businesses or Shops:
1. Highlight an Employee of the Week: Create a reel featuring your employee of the week, sharing their role in the business and their personality. Include shoutouts and accolades to create a more personal and engaging connection with your audience.
2. Behind-the-Scenes Content: Share your business's behind-the-scenes footage to allow your viewers to see what goes on behind closed doors. This humanizes your brand, enhances transparency, and helps build trust with your audience.
3. Product Reviews: Share reels where you post-test different products available in your shop and share your genuine opinions about them. Encourage followers to ask questions about your reviews to increase engagement and build authenticity.
4. How-to Videos: To demonstrate your product's usage or share related lifestyle tips, create educational or entertaining how-to videos. Make sure to insert links or swipe-up features for users who want to learn more or make a purchase.
5. Shop the Look: Use Instagram's branded content feature to create shoppable reels, where viewers can easily discover and purchase the products being promoted. Use complementary product suggestions to educate viewers about full outfit or style possibilities.
6. Partner with Your Local Community or Influencers: Collab with local organizations or influencers, sharing their promotions or social impact stories. This will help expand your reach to new customers interested in supporting your community.
7. Giveaways and Contests: Run giveaways or contests with business clothing line, discount codes, or complimentary products. Encourage users to participate, engage, and boost brand visibility and sentiment.
8. User-Generated Content: Encourage customers to share their experiences using your products by featuring their content in your Instagram Reels. This kind of content not only entices new followers but also builds credibility and trust.
9. Leverage Trending Hashtags: Stay on top of the latest hashtags and incorporate them into your content, especially those related to local events, holidays, or trends. This can help increase the reach and engagement of your posts.
10. Use High-Quality Visuals: Showcase your products, the environment, and the essence of your business using high-quality and eye-catching visuals. Make your images and videos vibrant and memorable to encourage users to want to engage with your posts.'''),
] 

response = chat.invoke(messages)
print(response.content)