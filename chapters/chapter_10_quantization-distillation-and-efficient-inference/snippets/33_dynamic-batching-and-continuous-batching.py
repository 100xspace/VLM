if not batch:
            return
        # Remove processed requests from queue
        for req in batch:
            self.request_queue.remove(req)
        # Prepare batched inputs
        images_batch = torch.cat([r["images"] for r in batch], dim=0
# Handle variable-length sequences per request; assumes each r["input_ids"] is [1, seq_len]
        seqs = [r["input_ids"][0] for r in batch]
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0).unsqueeze(0)
        # Run inference
        with torch.no_grad():
            outputs = self.model(images_batch, input_ids_batch)
        # Return results via callbacks
        for i, req in enumerate(batch):
            req["callback"](outputs[i])
    def run(self):
        """Continuous processing loop"""
        while True:
            self.process_batch()
            time.sleep(0.01)  # Avoid busy-waiting
# Usage example
# scheduler = DynamicBatchScheduler(model, max_batch_size=16)
# scheduler.add_request(image1, text1, callback1)
# scheduler.add_request(image2, text2, callback2)
# scheduler.run()
