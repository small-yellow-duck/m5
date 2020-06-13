
TODO:
- speed up data loader (merge operation is slow, probably there is a better way, maybe reindex data using the day...)
- set up the dataloader to use parallel processing
- the loss function does not match to the loss metric for the comp
- make model bidirectional (or figure out some scheme so that algo has some way to know "in a week it will be Easter/Christmas")
- randomize the start and end times for each training batch (need to write a custom DataLoader class)
- bias the training batches so that the validation sets are sampled from the same period in the leaderboard, but 
one year earlier
- make embeddings for the categorical values ['event_name_1_num', 'event_type_1_num', 'event_name_2_num', 'event_type_2_num'] 
that can be reused by both the enc and the dec

