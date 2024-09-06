# Hello everyone 👋
- this module ca help your large language model storage and manage your memories
## let's you view example code to understand how the way it active
- example code
```
from example_llm import ExampleLLM
from llm_memories_manager import WriteNewLocalMemories, ShortContextManager, ActivateLocalMemories

# example AI model
example_llm = ExampleLLM()
# create instance of memories manager methods
local_mem_writer = WriteNewLocalMemories(s="Human's say memories: ", t="Bot's say memories: ")
short_context_manager = ShortContextManager(s="Human's context say: ", t="Bot's context say: ")
activate_local_memories = ActivateLocalMemories()

user_input = input("You: ")

# activate local memories on user input
local_memories_activated = \
    activate_local_memories.activate_localMemories(user_input) # data type: list mem
# activate the context storage
short_context_activated = \
    short_context_manager.activate_shortContetxMem() # data type: None or list


# then you can concat local_memories_activated and short_context_activated to be input put in ll model
llm_input = {"\n".join(local_memories_activated)} + "\n" +\
    "\n".join(short_context_activated) if short_context_activated is not None else "" +\
    "\n\n" + f"current user input: {user_input}"
# output from model's predictions
llm_output = example_llm.predict(llm_input)

# then you can save these input and output in this time step in database of memories
local_mem_writer.write_newLocalMem(user_input, llm_output)
short_context_manager.save_to_shortContextMem(user_input, llm_output)

# complete, so then you can view output via print
print(llm_output)
```

## thank you for visit my project
**Let's support me via social media if you can. Thank you**
- my facebook: https://www.facebook.com/profile.php?id=61562099241369
- my youtube: https://www.youtube.com/@phucoding286
