from chunked_engine import Engine
from chunked_scheduler import Scheduler, InputRequest
engine = Engine()
scheduler = Scheduler(engine, token_batch_size=1024)

sample_prompts = ["Today is a rainy day"] * 1024 + ["UW is"] * 1024

# Enqueue and run
for prompt in sample_prompts:
    scheduler.add_req(InputRequest(prompt, output_len=100))
    # scheduler.run()

# Drain remaining requests
while not scheduler.finished():
    scheduler.run()

# scheduler.print_completed()
