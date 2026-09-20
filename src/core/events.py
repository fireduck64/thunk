import asyncio
from typing import Callable, Dict, Any, List

class EventBus:
    """
    A simple asynchronous Pub/Sub event bus.
    Allows components to subscribe to events (e.g. 'agent_started', 'tool_executed').
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    async def publish(self, event_type: str, data: Dict[str, Any] = None):
        if data is None:
            data = {}
        
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
