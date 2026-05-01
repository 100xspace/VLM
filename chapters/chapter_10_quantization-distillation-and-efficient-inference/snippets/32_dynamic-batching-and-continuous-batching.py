def __init__(self, model, max_batch_size=8, max_wait_time=0.1):
        self.model, self.max_batch_size, self.max_wait_time = model, max_batch_size, max_wait_time
        self.request_queue = []
    def add_request(self, images, input_ids, callback):
        """Add inference request to queue"""
        self.request_queue.append({"images": images, "input_ids": input_ids,
                                   "callback": callback, "timestamp": time.time()})
    def process_batch(self):
        """Process a dynamically formed batch"""
        if not self.request_queue:
            return
        # Sort by sequence length so we can group similar requests
        self.request_queue.sort(key=lambda r: r["input_ids"].shape[1])
        batch, current_length = [], None
        for req in self.request_queue[:self.max_batch_size]:
            seq_len = req["input_ids"].shape[1]
            # Group requests within ~20% length similarity to reduce padding waste
            if current_length is None or abs(seq_len - current_length) / current_length < 0.2:
                batch.append(req); current_length = seq_len
            # Stop if batch is full or oldest request waited too long
            if len(batch) >= self.max_batch_size or (time.time() - req["timestamp"] > self.max_wait_time):
