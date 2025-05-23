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
 * Routes the response from the model to handle different types of responses:
 * 1. AIMessage with tool calls
 * 2. AIMessage with direct content
 * 3. Final response from the model
 * 4. Tool call in old format
 */
async function routeResponse(response: any): Promise<string> {
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
      
      // Format the result based on the tool type
      if (toolCall.name === "getCurrentTemperature") {
        return `A temperatura atual é ${result}`;
      } else if (toolCall.name === "searchWikipedia") {
        return `Informações encontradas:\n${result}`;
      }
      
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
      
      // Format the result based on the tool type
      if (response.tool === "getCurrentTemperature") {
        return `A temperatura atual é ${result}`;
      } else if (response.tool === "searchWikipedia") {
        return `Informações encontradas:\n${result}`;
      }
      
      return String(result);
    } catch (error) {
      throw new Error(`Error executing tool ${response.tool}: ${error}`);
    }
  }

  // Case 5: Unexpected response format
  throw new Error(`Unexpected response format: ${JSON.stringify(response)}`);
}

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

    const result = await routeResponse(response);
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