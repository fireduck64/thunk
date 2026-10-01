---
summary: Use when you want to read a book from the library, managing the process using tasks and memory.
---

# Reading a Book

Reading a book is a long-term process that requires structured planning, as you cannot read the entire text in a single turn. Follow these steps to read and synthesize a book effectively:

1. **Find a Book:** Use `list_library` to see what books are available in the `/vault/ebook` directory.
2. **Start the Book:** Use `start_book` with the filename to begin. This resets your reading progress for that file.
3. **Plan the Reading:** Do not try to read without a plan. Use the `push_task` tool to create a sequence of manageable reading tasks in your Task Queue. 
   * Example task: "Read 'Mr. Darcy' chunks 1-10 and synthesize."
   * Example task: "Read 'Mr. Darcy' chunks 11-20 and synthesize."
4. **Read and Synthesize:** 
   * For your current task, call `read_chunk` multiple times to consume the text.
   * Actively analyze themes, lore, and character arcs as you read.
5. **Store the Knowledge:** 
   * Use `save_semantic_memory` to permanently store your summaries and insights so you can conceptually recall them later.
   * Use `put_note` strictly for tracking your exact structural state (e.g., key: `reading_status`, value: `Currently on chunk 10 of Mr. Darcy`).
6. **Iterate:** Complete your current task, use `pop_task` to get the next chunk, and continue the cycle until the book is finished.