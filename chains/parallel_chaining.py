from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "MiniMaxAI/MiniMax-M1-80k",
    max_new_tokens = 512,
    temperature =0.5,
)

model = ChatHuggingFace(llm=llm)

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a movie critic."),
        ("human","Provide me a brief summary of the movie {movie_name}"),
    ]
)

def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic."),
            ("human","Analyze the plot: {plot}. what are its strength and weaknesses ?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)


def analyze_character(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system","You are a movie critic."),
            ("human","Analyze the characters: {characters}. what are their stengths and weaknesses?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

# combine analysis and final verdict

def combine_verdicts(plot_analysis, charcter_analysis):
    return f"Plot analysis: {plot_analysis}\n\n Character Analysis:\n{charcter_analysis}"



# simplyfy branches with lecl

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x) ) | model | StrOutputParser()
)



character_branch_chain = (
    RunnableLambda(lambda x: analyze_character(x)) | model | StrOutputParser()
)



# create combined chain using lecl

chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches= {"plot": plot_branch_chain, "characters" : character_branch_chain})
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]['plot'], x["branches"]["characters"]))
    )

result = chain.invoke({"movie_name": "Inception"})

print(result)


# otuput 

'''

Plot analysis: <think>
Okay, I need to analyze the strengths and weaknesses of the movie Inception. Let's start by recalling what I know about the film. Inception is a sci-fi thriller directed by Christopher Nolan, known for its complex plot and layered storytelling. The user wants a brief summary first, which I provided, and now they want the strengths and weaknesses.

First, strengths. The movie is praised for its innovative concept of dreams within dreams, the visual effects, and the direction. The plot is intricate but well-structured, keeping the audience engaged. The performances, especially DiCaprio, are strong. The themes of reality vs. dreams, guilt, and memory are deep. The action sequences, like the hallway fight scene, are memorable. The score by Hans Zimmer is also a strength, adding tension.

Weaknesses might include the complexity of the plot, which some viewers find confusing. The exposition is heavy at times, with characters explaining the rules of the dream world, which could feel too much. Some characters might be underdeveloped, like Ariadne or Arthur, compared to Cobb. The emotional depth of Cobb's guilt is a strength, but maybe some find the emotional arcs less impactful compared to the plot. Also, the ending being ambiguous could be a weakness if viewers prefer clear resolutions.

Wait, but the user asked for strengths and weaknesses. Let me list them clearly. Strengths: innovative concept, visual effects, direction, performances, themes, score. Weaknesses: complex plot might confuse some, heavy exposition, underdeveloped secondary characters, ambiguous ending might not satisfy everyone.

I should structure this into coherent points. Maybe start with strengths, then weaknesses. Make sure to mention specific examples, like the dream layers, the hallway scene, the use of music. For weaknesses, maybe the exposition scenes where they explain the rules, or characters like Arthur not having much depth. Also, the emotional core is Cobb's guilt, which is a strength, but maybe the supporting characters' arcs are weaker.

Also, check if there are other weaknesses. Some critics say the movie is too long, or that the action overshadows the emotional aspects. But Inception is generally well-received, so the weaknesses are minor compared to strengths. Need to balance both.

So, summarizing:

Strengths:
- Innovative and complex plot structure with layered dreams.
- Visually stunning effects (e.g., folding city, hallway fight).
- Strong direction by Nolan, maintaining tension and coherence.
- Memorable performances, especially DiCaprio and Cotillard.
- Hans Zimmer's score enhances the atmosphere.
- Thematic depth on reality, guilt, and memory.

Weaknesses:
- Overly complex plot may confuse some viewers.
- Heavy exposition at times to explain dream rules.
- Some secondary characters (e.g., Arthur, Ariadne) are underdeveloped.
- Ambiguous ending might frustrate those seeking clarity.

I should present this in a clear, concise way, maybe bullet points or paragraphs. The user probably wants a balanced analysis, so highlight both aspects with examples.</think>     

**Strengths of *Inception*:**
1. **Innovative Concept & Structure**: The layered dreamscapes and time-dilation mechanics create a unique, mind-bending narrative. The heist framework applied to subconscious exploration is fresh and intellectually engaging.
2. **Visual Spectacle**: Iconic sequences (e.g., the folding Paris streets, zero-gravity hallway fight) showcase groundbreaking CGI and practical effects, enhancing the surreal dream logic.
3. **Direction & Pacing**: Christopher Nolan balances complexity with clarity, maintaining tension across multiple dream layers. The editing and pacing keep the audience invested despite the intricate plot.
4. **Performances**: Leonardo DiCaprio anchors the film with a nuanced portrayal of Cobb’s guilt, while supporting roles (e.g., Tom Hardy’s Eames, Marion Cotillard’s Mal) add depth and intrigue.
5. **Thematic Depth**: Explores themes of memory, guilt, and reality vs. illusion, inviting philosophical reflection. Cobb’s emotional arc (his grief over Mal) adds emotional weight to the cerebral plot.
6. **Score**: Hans Zimmer’s pulsating soundtrack amplifies tension and wonder, becoming integral to the film’s identity.

**Weaknesses of *Inception*:**
1. **Complexity Overload**: The layered plot risks alienating viewers who prefer straightforward storytelling. Exposition-heavy scenes (e.g., Ariadne’s “dream rules” tutorial) can feel didactic.
2. **Underdeveloped Characters**: Secondary characters (Arthur, Ariadne) lack depth compared to Cobb, serving functional roles rather than emotional arcs.
3. **Ambiguous Ending**: While intentional, the spinning top’s unresolved fate may frustrate viewers seeking closure.
4. **Emotional Balance**: The focus on plot mechanics occasionally overshadows the emotional core (Cobb’s guilt), making some relationships feel secondary.

**Verdict**: *Inception* excels as a visually stunning, intellectually stimulating thriller but occasionally prioritizes complexity over character depth. Its strengths lie in its originality and execution, while its weaknesses stem from its ambition to balance narrative density with emotional resonance.

 Character Analysis:
<think>
Okay, let's tackle this. The user wants a brief summary of Inception, but now they're asking about the strengths and weaknesses of the characters. Wait, the original query was to analyze the characters, but the user provided a summary. Maybe they want a character analysis now?

Wait, the initial user message was "Analyze the characters: <think> [summary]". So the user wants an analysis of the characters in Inception. The previous response was a summary, but the actual task is to analyze the characters. Let me check the history.

The user wrote: "Analyze the characters: <think> [summary]". So they provided a summary and want an analysis of the characters. The assistant's previous response was the summary, but the user's actual query is to analyze the characters. Maybe the user pasted the summary into the think tag and wants the analysis now.

So the task is to analyze the characters of Inception. Let's list the main characters and their strengths and weaknesses.

Main characters: Cobb, Arthur, Ariadne, Eames, Saito, Robert Fischer, Mal (as a projection), maybe others like Peter Browning.

Cobb: Strengths – skilled extractor, determined, loyal to his team. Weaknesses – guilt over Mal, which clouds his judgment, emotionally unstable.

Arthur: Cobb's partner. Strengths – experienced, level-headed. Weaknesses – maybe too reliant on Cobb, less creative?

Ariadne: Young architect. Strengths – quick learner, creative. Weaknesses – inexperienced, maybe naive.

Eames: Forger. Strengths – adaptable, charismatic, good under pressure. Weaknesses – maybe reckless, likes to improvise.

Saito: Client. Strengths – resourceful, persistent. Weaknesses – maybe manipulative, has his own agenda.

Robert Fischer: Target. Strengths – determined to prove himself. Weaknesses – conflicted, vulnerable to manipulation.

Mal: Projection. Strengths – persistent, powerful in the dream. Weaknesses – represents Cobb's guilt, a liability.

Need to structure this clearly, maybe each character with strengths and weaknesses. Also mention how their traits affect the plot.

Wait, the user might want a more detailed analysis, perhaps their roles in the story and how their traits contribute to the mission's success or challenges.

Also, consider themes: reality vs. dreams, guilt, etc., but the question is about characters' strengths and weaknesses.

Let me outline each character:

**Cobb (Dom Cobb)**
- *Strengths*: Highly skilled extractor, knowledgeable about dream mechanics, determined, protective of his team.
- *Weaknesses*: Haunted by guilt over Mal’s death, which manifests as projections disrupting missions; emotionally unstable, risks the mission for personal closure.

**Arthur (Cobb’s Partner)**
- *Strengths*: Experienced, pragmatic, loyal to Cobb.
- *Weaknesses*: Less innovative than Cobb, struggles with adapting to unpredictable dream layers (e.g., zero-gravity fight in the hotel).

**Ariadne (Young Architect)**
- *Strengths*: Quick learner, creative in designing dream mazes, provides fresh perspective.
- *Weaknesses*: Inexperienced, initially naive about the dangers of shared dreaming.

**Eames (Forger)**
- *Strengths*: Charismatic, adaptable, skilled at impersonation and improvisation (e.g., pretending to be Peter Browning).
- *Weaknesses*: Overconfident, takes risks (e.g., provoking projections in the third layer).

**Saito (Client)**
- *Strengths*: Wealthy, resourceful, persistent in achieving his goal.
- *Weaknesses*: Manipulative, withholds information (e.g., Cobb’s fugitive status), prioritizes his agenda over the team’s safety.

**Robert Fischer (Target)**
- *Strengths*: Determined to forge his own path, resilient in facing subconscious defenses.
- *Weaknesses*: Vulnerable to manipulation, conflicted about his father’s legacy.

**Mal (Projection)**
- *Strengths*: Powerful, persistent, embodies Cobb’s guilt.
- *Weaknesses*: A manifestation of Cobb’s psyche, ultimately a liability that threatens the mission.

**Peter Browning (Peter’s Uncle)**
- *Strengths*: Cunning, exploits Robert’s insecurities.
- *Weaknesses*: Overly aggressive, underestimates the team’s planning.

Themes: Cobb’s guilt and redemption, the fragility of reality, teamwork vs. individual flaws.

Need to present this clearly, maybe in bullet points or paragraphs. Ensure each character’s role and traits are connected to the plot’s outcome.</think>

### Character Analysis of *Inception* (2010)

#### **Dom Cobb (Leonardo DiCaprio)**
- **Strengths**:
  - A master extractor with unmatched expertise in dream mechanics, Cobb is strategic, resourceful, and deeply committed to his team.
  - His determination to reunite with his children drives him to take risks, showcasing resilience.
- **Weaknesses**:
  - Haunted by guilt over his wife Mal’s death, Cobb’s subconscious projections (Mal) repeatedly sabotage missions.
  - His emotional instability (e.g., refusing to let go of Mal’s memory) jeopardizes the team’s safety and clarity.

#### **Arthur (Joseph Gordon-Levitt)**
- **Strengths**:
  - Cobb’s pragmatic partner, Arthur is level-headed, experienced, and loyal. He anchors the team during chaotic moments.
- **Weaknesses**:
  - Less adaptable than Cobb; struggles with unexpected dream-layer shifts (e.g., the zero-gravity fight in the hotel layer).
  - Relies heavily on Cobb’s leadership, limiting his problem-solving autonomy.

#### **Ariadne (Ellen Page)**
- **Strengths**:
  - A brilliant, quick-learning architect, Ariadne designs intricate dream mazes (e.g., the rotating hallway) and provides fresh perspective.
  - Her curiosity and moral compass keep the team grounded (e.g., questioning Cobb’s motives).
- **Weaknesses**:
  - Inexperienced in shared dreaming, she initially underestimates the dangers (e.g., nearly getting lost in limbo).
  - Naive about the psychological toll of the mission.

#### **Eames (Tom Hardy)**
- **Strengths**:
  - A charismatic forger, Eames excels at improvisation (e.g., impersonating Peter Browning) and thrives under pressure.
  - His humor and confidence boost team morale.
- **Weaknesses**:
  - Overconfident and reckless (e.g., provoking projections in the third layer), he risks exposing the team.
  - Prioritizes personal flair over caution.

#### **Saito (Ken Watanabe)**
- **Strengths**:
  - Wealthy and resourceful, Saito funds the mission and leverages his influence to protect Cobb post-mission.
  - Persistent in achieving his goal (reviving his company).
- **Weaknesses**:
  - Manipulative, withholding key information (e.g., Cobb’s fugitive status) to secure Cobb’s cooperation.
  - Prioritizes his agenda over the team’s safety (e.g., pushing for a risky three-layer dive).

#### **Robert Fischer (Cillian Murphy)**
- **Strengths**:
  - Determined to escape his father’s shadow, Robert shows resilience in confronting his subconscious (e.g., battling projections).
  - His vulnerability makes him a compelling target for inception.
- **Weaknesses**:
  - Emotionally conflicted, he’s easily manipulated by projections (e.g., his guilt over his father’s disapproval).
  - Initially distrustful of the team, he resists cooperation.

#### **Mal (Marion Cotillard)**
- **Strengths**:
  - A powerful projection of Cobb’s guilt, Mal is relentless and embodies his deepest fears.
  - Her presence raises stakes, testing Cobb’s emotional resolve.
- **Weaknesses**:
  - A manifestation of Cobb’s psyche, she is not a "real" antagonist but a liability that threatens the mission’s success.

#### **Peter Browning (Tom Berenger)**
- **Strengths**:
  - Cunning and manipulative, he exploits Robert’s insecurities to protect his own interests.
- **Weaknesses**:
  - Overly aggressive, he underestimates the team’s planning (e.g., falling for Eames’ impersonation).

### Key Themes & Impact
- **Cobb’s Guilt**: His unresolved trauma over Mal drives both his strengths (determination) and weaknesses (self-sabotage).
- **Team Dynamics**: The mission’s success hinges on balancing individual strengths (Ariadne’s creativity, Eames’ improvisation) against flaws (Cobb’s instability, Arthur’s rigidity).
- **Reality vs. Dreams**: Characters’ traits (e.g., Cobb’s guilt, Robert’s vulnerability) blur the line between subconscious projections and reality, heightening tension.

In *Inception*, characters’ strengths and weaknesses intertwine to create a high-stakes, emotionally charged heist, where personal demons and teamwork collide.

'''