/**
 * Chat Models with Tools
 * 
 * This example demonstrates how to use tools with chat models in LangChain,
 * including creating custom tools, routing responses, and handling tool execution.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { config } from "dotenv";
import { Logger } from "./Logger";
import axios from "axios";

// Load environment variables
config();

/**
 * Tool Schemas
 */
const TemperatureArgsSchema = z.object({
  latitude: z.number().describe("Latitude of the location to get temperature"),
  longitude: z.number().describe("Longitude of the location to get temperature"),
});

type TemperatureArgs = z.infer<typeof TemperatureArgsSchema>;

/**
 * Custom Tools Implementation
 */

// Tool to get current temperature
const getCurrentTemperature = tool(
  async (input: TemperatureArgs): Promise<number> => {
    try {
      const URL = "https://api.open-meteo.com/v1/forecast";
      const params = {
        latitude: input.latitude,
        longitude: input.longitude,
        hourly: "temperature_2m",
        forecast_days: 1,
      };

      const response = await axios.get(URL, { params });
      
      if (response.status === 200) {
        const result = response.data;
        const now = new Date();
        const hours = result.hourly.time.map((timeStr: string) => new Date(timeStr));
        
        // Find closest hour to current time
        const closestHourIndex = hours.reduce((closest: number, hour: Date, index: number) => {
          const currentDiff = Math.abs(hour.getTime() - now.getTime());
          const closestDiff = Math.abs(hours[closest].getTime() - now.getTime());
          return currentDiff < closestDiff ? index : closest;
        }, 0);

        return result.hourly.temperature_2m[closestHourIndex];
      } else {
        throw new Error(`Request to API ${URL} failed: ${response.status}`);
      }
    } catch (error) {
      Logger.error(`Error getting temperature: ${error}`);
      throw error;
    }
  },
  {
    name: "getCurrentTemperature",
    description: "Gets the current temperature for given coordinates",
    schema: TemperatureArgsSchema,
  }
);

// Tool to search Wikipedia
const searchWikipedia = tool(
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

/**
 * Chain Setup
 */
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a friendly assistant named Isaac"],
  ["user", "{input}"],
]);

const chat = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});

const tools = [getCurrentTemperature, searchWikipedia];

/**
 * Routing Function
 * 
 * This function handles the routing of responses from the model,
 * determining whether to execute a tool or return the final response.
 */
async function routeResponse(response: any): Promise<string> {
  Logger.debug(`Routing response: ${JSON.stringify(response, null, 2)}\n`);

  // Case 1: AIMessage with tool calls
  if (response.tool_calls && response.tool_calls.length > 0) {
    Logger.info(`Processing tool call from AIMessage\n`);
    const toolCall = response.tool_calls[0];
    const toolToRun = tools.find((tool) => tool.name === toolCall.name);
    
    if (!toolToRun) {
      throw new Error(`Tool ${toolCall.name} not found. Available tools: ${tools.map(t => t.name).join(', ')}`);
    }
    
    try {
      Logger.debug(`Executing tool ${toolCall.name} with input: ${JSON.stringify(toolCall.args)}\n`);
      const result = await (toolToRun as any)._call(toolCall.args);
      return String(result);
    } catch (error) {
      throw new Error(`Error executing tool ${toolCall.name}: ${error}`);
    }
  }
  
  // Case 2: AIMessage with direct content
  else if (response.content) {
    Logger.info("Processing AIMessage response\n");
    return response.content;
  }
  
  // Case 3: Final response from the model
  else if (response.returnValues) {
    Logger.info("Processing final response from model\n");
    return response.returnValues.output;
  } 
  
  // Case 4: Tool call in old format
  else if (response.tool && response.toolInput) {
    Logger.info(`Processing tool call for ${response.tool}\n`);
    const toolToRun = tools.find((tool) => tool.name === response.tool);
    if (!toolToRun) {
      throw new Error(`Tool ${response.tool} not found. Available tools: ${tools.map(t => t.name).join(', ')}`);
    }
    try {
      Logger.debug(`Executing tool ${response.tool} with input: ${JSON.stringify(response.toolInput)}\n`);
      const result = await (toolToRun as any)._call(response.toolInput);
      return String(result);
    } catch (error) {
      throw new Error(`Error executing tool ${response.tool}: ${error}`);
    }
  }

  // Case 5: Unexpected response format
  throw new Error(`Unexpected response format: ${JSON.stringify(response)}`);
}

/**
 * Main Chain
 */
const chain = RunnableSequence.from([
  prompt,
  chat.bind({ tools }),
  routeResponse,
]);

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting chat models with tools example\n");

    // Test simple greeting
    Logger.info("Testing simple greeting\n");
    const greetingResponse = await chain.invoke({
      input: "Hello",
    });
    Logger.success("Greeting response received\n");
    Logger.debug(`Greeting response: ${greetingResponse}\n`);

    // Test Wikipedia search
    Logger.info("Testing Wikipedia search\n");
    const wikiResponse = await chain.invoke({
      input: "Who was Isaac Asimov?",
    });
    Logger.success("Wikipedia response received\n");
    Logger.debug(`Wikipedia response: ${wikiResponse}\n`);

    // Test temperature check
    Logger.info("Testing temperature check\n");
    const tempResponse = await chain.invoke({
      input: "What's the temperature in São Paulo?",
    });
    Logger.success("Temperature response received\n");
    Logger.debug(`Temperature response: ${tempResponse}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 