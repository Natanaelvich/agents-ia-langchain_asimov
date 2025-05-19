export class Logger {
  private static readonly colors = {
    reset: "\x1b[0m",
    bright: "\x1b[1m",
    dim: "\x1b[2m",
    underscore: "\x1b[4m",
    blink: "\x1b[5m",
    reverse: "\x1b[7m",
    hidden: "\x1b[8m",
    
    fg: {
      black: "\x1b[30m",
      red: "\x1b[31m",
      green: "\x1b[32m",
      yellow: "\x1b[33m",
      blue: "\x1b[34m",
      magenta: "\x1b[35m",
      cyan: "\x1b[36m",
      white: "\x1b[37m",
    },
    
    bg: {
      black: "\x1b[40m",
      red: "\x1b[41m",
      green: "\x1b[42m",
      yellow: "\x1b[43m",
      blue: "\x1b[44m",
      magenta: "\x1b[45m",
      cyan: "\x1b[46m",
      white: "\x1b[47m",
    }
  };

  static info(message: string): void {
    console.log(`${this.colors.fg.cyan}[INFO]${this.colors.reset} ${message}`);
  }

  static success(message: string): void {
    console.log(`${this.colors.fg.green}[SUCCESS]${this.colors.reset} ${message}`);
  }

  static error(message: string): void {
    console.error(`${this.colors.fg.red}[ERROR]${this.colors.reset} ${message}`);
  }

  static warning(message: string): void {
    console.warn(`${this.colors.fg.yellow}[WARNING]${this.colors.reset} ${message}`);
  }

  static debug(message: string): void {
    console.log(`${this.colors.fg.magenta}[DEBUG]${this.colors.reset} ${message}`);
  }

  static chain(message: string): void {
    console.log(`${this.colors.fg.blue}[CHAIN]${this.colors.reset} ${message}`);
  }
} 