# JARVIS Voice Assistant

class JARVIS:
    def __init__(self):
        self.commands = {
            'hello': self.say_hello,
            'goodbye': self.say_goodbye,
            'time': self.tell_time,
        }

    def run(self):
        while True:
            command = input('How can I assist you? ').lower()
            if command in self.commands:
                self.commands[command]()
            else:
                print('Sorry, I did not understand that command.')

    def say_hello(self):
        print('Hello! I am your voice assistant.')

    def say_goodbye(self):
        print('Goodbye! Have a great day!')

    def tell_time(self):
        from datetime import datetime
        now = datetime.now()
        print(f'The current time is {now.strftime("%H:%M:%S")}')

if __name__ == '__main__':
    jarvis = JARVIS()
    jarvis.run()