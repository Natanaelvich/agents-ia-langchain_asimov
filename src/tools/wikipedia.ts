import { tool } from "@langchain/core/tools";
import { z } from "zod";
import axios from "axios";
import { Logger } from "../Logger";

/**
 * Tool to search Wikipedia
 */
export const searchWikipedia = tool(
  async (input: { query: string }): Promise<string> => {
    try {
      const response = await axios.get("https://pt.wikipedia.org/w/api.php", {
        params: {
          action: "query",
          list: "search",
          srsearch: input.query,
          format: "json",
          origin: "*",
        },
      });

      if (response.status === 200) {
        const searchResults = response.data.query.search.slice(0, 3);
        const summaries = [];

        for (const result of searchResults) {
          const pageResponse = await axios.get("https://pt.wikipedia.org/w/api.php", {
            params: {
              action: "query",
              prop: "extracts",
              exintro: true,
              titles: result.title,
              format: "json",
              origin: "*",
            },
          });

          if (pageResponse.status === 200) {
            const pages = pageResponse.data.query.pages;
            const pageId = Object.keys(pages)[0];
            const summary = pages[pageId].extract;
            summaries.push(`Title: ${result.title}\nSummary: ${summary}`);
          }
        }

        return summaries.length > 0 ? summaries.join("\n\n") : "No results found";
      } else {
        throw new Error("Wikipedia search failed");
      }
    } catch (error) {
      Logger.error(`Error searching Wikipedia: ${error}`);
      throw error;
    }
  },
  {
    name: "searchWikipedia",
    description: "Searches Wikipedia and returns summaries of pages for the query",
    schema: z.object({
      query: z.string().describe("The search query"),
    }),
  }
); 