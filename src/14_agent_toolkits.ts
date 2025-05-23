/**
 * Agent Toolkits
 * 
 * This example demonstrates how to use agent toolkits in LangChain:
 * 1. Pandas DataFrame Toolkit - For data analysis and manipulation
 * 2. SQL Database Toolkit - For database operations and queries
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { config } from "dotenv";
import { Logger } from "./Logger";
import { MessagesPlaceholder } from "@langchain/core/prompts";
import { AgentAction, AgentFinish } from "@langchain/core/agents";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { AIMessage, FunctionMessage } from "@langchain/core/messages";
import { dfTools } from "./tools/dataframe";
import { sqlTools } from "./tools/sql";

// Load environment variables
config();

/**
 * Agent Setup
 * 
 * The prompt template defines how we want to structure our conversation with the model.
 * It includes:
 * - A system message that sets the context
 * - A user message that contains the input
 * - A MessagesPlaceholder for the agent's scratchpad (where it keeps track of its reasoning)
 */
const dfPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a friendly assistant named Isaac. You help users analyze data using pandas DataFrames."],
  ["user", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const sqlPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a friendly assistant named Isaac. You help users query databases using SQL."],
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
 * RunnablePassthrough is a utility that allows us to:
 * 1. Pass through certain inputs unchanged
 * 2. Transform other inputs as needed
 * 3. Add new fields to the input
 * 
 * In our case, we use it to transform the intermediate steps into a format
 * that can be used in the agent's scratchpad.
 */
interface AgentInput {
  input: string;
  intermediate_steps: [AgentAction, string][];
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

/**
 * RunnableSequence is used to create a chain of operations that will be executed in sequence.
 * Each step in the sequence takes the output of the previous step as its input.
 * 
 * Our sequence:
 * 1. passThrough: Transforms the input and adds the agent_scratchpad
 * 2. prompt: Formats the messages according to our template
 * 3. chat.bind: Sends the messages to the model with our tools
 */
const dfChain = RunnableSequence.from([
  passThrough,
  dfPrompt,
  chat.bind({ tools: dfTools }),
]);

const sqlChain = RunnableSequence.from([
  passThrough,
  sqlPrompt,
  chat.bind({ tools: sqlTools }),
]);

/**
 * Routes the response from the model to handle different types of responses:
 * 1. AIMessage with tool calls
 * 2. AIMessage with direct content
 * 3. Final response from the model
 * 4. Tool call in old format
 */
async function routeResponse(response: any, tools: any[]): Promise<string> {
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
 * Agent Executor
 * 
 * The runAgent function implements the agent's execution loop:
 * 1. Get a response from the model
 * 2. Route the response to handle different types
 * 3. Add the result to the intermediate steps
 * 4. Repeat until we get a final response
 */
async function runDataFrameAgent(input: string) {
  let intermediateSteps: [AgentAction, string][] = [];

  while (true) {
    const response = await dfChain.invoke({
      input,
      intermediate_steps: intermediateSteps,
    });

    // If we got a direct response with content, return it
    if (response.content && !response.tool_calls?.length) {
      return response.content;
    }

    // If we got a final response with returnValues, return it
    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
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

    const result = await routeResponse(response, dfTools);
    intermediateSteps.push([action, result]);
  }
}

async function runSQLAgent(input: string) {
  let intermediateSteps: [AgentAction, string][] = [];

  while (true) {
    const response = await sqlChain.invoke({
      input,
      intermediate_steps: intermediateSteps,
    });

    // If we got a direct response with content, return it
    if (response.content && !response.tool_calls?.length) {
      return response.content;
    }

    // If we got a final response with returnValues, return it
    if ('returnValues' in response && typeof response.returnValues === 'object' && response.returnValues !== null) {
      const returnValues = response.returnValues as { output: string };
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

    const result = await routeResponse(response, sqlTools);
    intermediateSteps.push([action, result]);
  }
}

/**
 * Example Usage
 */
async function main() {
  try {
    Logger.info("Starting agent toolkits example\n");

    // Test DataFrame agent
    Logger.info("Testing DataFrame agent\n");
    const dfResponse = await runDataFrameAgent("What's the shape of the DataFrame?");
    Logger.success("DataFrame response received\n");
    Logger.debug(`DataFrame response: ${dfResponse}\n`);

    // Test SQL agent
    Logger.info("Testing SQL agent\n");
    const sqlResponse = await runSQLAgent("List all tables in the database");
    Logger.success("SQL response received\n");
    Logger.debug(`SQL response: ${sqlResponse}\n`);

  } catch (error) {
    Logger.error(`Error in main: ${error}\n`);
  }
}

// Run the example
main(); 