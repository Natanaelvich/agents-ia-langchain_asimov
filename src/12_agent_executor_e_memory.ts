/**
 * Agent Executor with Memory
 * 
 * This example demonstrates how to create an agent executor with memory capabilities,
 * allowing the agent to remember previous interactions and maintain context.
 * Uses FileSystemChatMessageHistory to persist chat history to the file system.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { config } from "dotenv";
import { Logger } from "./Logger";
import axios from "axios";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentAction, AgentFinish } from "@langchain/core/agents";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage, FunctionMessage, HumanMessage } from "@langchain/core/messages";
import { FileSystemChatMessageHistory } from "@langchain/community/stores/message/file_system";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { routeResponse } from "./utils/routeResponse";

// Load environment variables
config();

/**
 * Tool Schemas
 * 
 * We use Zod to define the schema for our tool inputs.
 * This provides type safety and validation for the tool parameters.
 */
const TemperatureArgsSchema = z.object({
  latitude: z.number().describe("Latitude of the location to get temperature"),
  longitude: z.number().describe("Longitude of the location to get temperature"),
});

type TemperatureArgs = z.infer<typeof TemperatureArgsSchema>;

/**
 * Custom Tools Implementation
 * 
 * The @tool decorator from LangChain is used to create tools that can be used by the agent.
 * It provides:
 * 1. Type safety through Zod schemas
 * 2. Automatic function calling capability
 * 3. Integration with the agent's reasoning process
 */

// Tool to get current temperature
const getCurrentTemperature = tool(
  async (input: TemperatureArgs): Promise<string> => {
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

        return `${result.hourly.temperature_2m[closestHourIndex]}°C`;
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
 * Agent Setup with Memory
 * 
 * We create a FileSystemChatMessageHistory instance to store chat history in the file system.
 * This provides persistence across sessions and better memory management.
 */
const getMessageHistory = async (sessionId: string) => {
  return new FileSystemChatMessageHistory({
    sessionId,
    userId: "user-id",
  });
};

const tools = [getCurrentTemperature, searchWikipedia];

/**
 * Agent Setup
 * 
 * The prompt template now includes a MessagesPlaceholder for chat history,
 * allowing the agent to access previous conversations.
 */
const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a friendly assistant named Isaac"],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const chat = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-0125",
  temperature: 0,
});

/**
 * Agent Chain
 * 
 * The chain now includes memory in its execution flow.
 */
interface AgentInput {
  input: string;
  intermediate_steps: [AgentAction, string][];
  chat_history?: (HumanMessage | AIMessage)[];
}

const passThrough = RunnablePassthrough.assign({
  agent_scratchpad: (input: Record<string, unknown>) => {
    const steps = (input.intermediate_steps as [AgentAction, string][]) || [];
    return steps.map(([action, observation]) => {
      const message = new AIMessage({
        content: "",
        additional_kwargs: {
          function_call: {
            name: action.tool,
            arguments: JSON.stringify(action.toolInput),
          },
        },
      });
      return message;
    });
  },
});

const chain = RunnableSequence.from([
  passThrough,
  prompt,
  chat.bind({ tools }),
]);

/**
 * Agent Executor with Memory
 * 
 * The runAgent function now uses FileSystemChatMessageHistory for persistent memory:
 * 1. Creates a new message history instance for the session
 * 2. Gets a response from the model
 * 3. Routes the response to handle different types
 * 4. Adds the result to the intermediate steps
 * 5. Saves the interaction to the file system
 * 6. Repeat until we get a final response
 */
async function runAgent(input: string, sessionId: string = "default-session") {
  let intermediateSteps: [AgentAction, string][] = [];
  
  // Get message history for the session
  const messageHistory = await getMessageHistory(sessionId);
  const chatHistory = await messageHistory.getMessages();

  while (true) {
    const response = await chain.invoke({
      input,
      intermediate_steps: intermediateSteps,
      chat_history: chatHistory,
    });

    // If we got a direct response with content, save it to history and return
    if (response.content && !response.tool_calls?.length) {
      await messageHistory.addMessage(new HumanMessage(input));
      await messageHistory.addMessage(new AIMessage(String(response.content)));
      return response.content;
    }

    // If we got a final response with returnValues, save it to history and return
    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
      await messageHistory.addMessage(new HumanMessage(input));
      await messageHistory.addMessage(new AIMessage(returnValues.output));
      return returnValues.output;
    }

    // Get tool name and args from the response
    const toolName = response.tool_calls?.[0]?.name;
    const toolArgs = response.tool_calls?.[0]?.args;

    // Validate that we have a valid tool call
    if (!toolName || !toolArgs) {
      throw new Error(`Invalid tool call response: ${JSON.stringify(response)}`);
    }

    // Create the agent action
    const action: AgentAction = {
      tool: toolName,
      toolInput: toolArgs,
      log: typeof response.content === 'string' ? response.content : ''
    };

    const result = await routeResponse(response, tools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting agent executor with file system memory example\n");

    // Test basic conversation
    Logger.info("Testing basic conversation\n");
    const response1 = await runAgent("Meu nome é Adriano", "test-session-1");
    Logger.success("Response received\n");
    Logger.debug(`Response: ${response1}\n`);

    // Test memory
    Logger.info("Testing memory\n");
    const response2 = await runAgent("Qual o meu nome?", "test-session-1");
    Logger.success("Response received\n");
    Logger.debug(`Response: ${response2}\n`);

    // Test temperature query
    Logger.info("Testing temperature query\n");
    const response3 = await runAgent("Qual é a temperatura em Porto Alegre?", "test-session-1");
    Logger.success("Response received\n");
    Logger.debug(`Response: ${response3}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 