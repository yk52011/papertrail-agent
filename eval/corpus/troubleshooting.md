# Failure recovery
For an out of memory error, reduce batch_size from 32 to 4 before retrying.
If tokenizer assets are missing, populate the tokenizer cache before offline execution.
