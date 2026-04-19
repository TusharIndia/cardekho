# Car Recommendation Chatbot

## What I Built And Why
I built a chatbot-first car recommendation app focused on best user experience. The goal was to make it easy for anyone to ask general car queries in natural language and get useful suggestions without learning a complex filter UI.

The system accepts user messages, extracts intent, and returns recommendations from the dataset with clear reasoning. It also handles budget-focused, family-size, and preference-focused requests in one flow.

## What I Deliberately Cut
- I did not build a full production-grade conversational memory system for long multi-turn context.
- I did not add a fully structured seating-capacity source because that field is not available in the current dataset.
- I avoided over-engineering the UI while core parsing and accuracy were still being improved.

These cuts were intentional to keep delivery fast and useful.

## Tech Stack And Why
- Flask:
  Lightweight, quick to iterate, and easy to deploy.
- Gemini API:
  Useful for natural language understanding and flexible intent extraction. Also practical because there is a free tier.
- Pandas:
  Fast and simple for CSV preprocessing and recommendation filtering.
- HTML, CSS, JavaScript:
  Lightweight chatbot interface with streaming-style response display.

## AI Tools vs Manual Work
### What I delegated to AI tools
- Large parts of code writing and refactoring
- Test harness generation and repeated evaluation runs
- Bulk test case expansion and structured coverage checks

### What I did manually
- Problem framing and reasoning about user behavior
- Deciding how parser logic should behave for edge cases
- Reviewing failures and choosing the final fix strategy
- Part of test analysis and result interpretation

## Where Tools Helped Most
- Bulk testing at scale with fast feedback loops
- Writing complex parser logic faster than manual-only coding
- Iterating quickly over many edge cases and message styles

## Where Tools Got In The Way
- Occasional overfitting toward specific examples
- Some generated logic was syntactically valid but semantically too broad
- A few edits needed manual correction to match intended behavior exactly

## If I Had Another 4 Hours
- Do deeper manual testing on long conversational flows
- Improve UI polish and clarity further
- Add stronger retrieval-based grounding so extreme robust queries get more consistently accurate answers
- Add richer dataset fields (for example true seating-capacity) to reduce heuristics

## Run Locally
1. Install dependencies.
2. Set gemini_api_key in .env.
3. Run script.py to regenerate cleaned data when needed.
4. Run app.py and open localhost.

## Project Files
- app.py: Flask backend and query handling
- script.py: dataset preprocessing
- query_test_cases.csv: evaluation queries
- evaluate_test_cases.py: automated evaluator
