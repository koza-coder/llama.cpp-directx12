<script lang="ts">
	// MemT native memory: choose the person (memory id) under the message box
	import { memoryStore } from '#lib/stores/memory.svelte.ts';
	import { onMount } from 'svelte';

	interface Props {
		// refresh the id list when a reply ends (a commit can add an id)
		isLoading?: boolean;
	}

	let { isLoading = false }: Props = $props();

	let draft = $state(memoryStore.id);

	onMount(() => {
		memoryStore.refresh();
	});

	$effect(() => {
		if (!isLoading) memoryStore.refresh();
	});

	function apply() {
		memoryStore.setId(draft);
		draft = memoryStore.id;
	}
</script>

{#if memoryStore.available}
	<div
		class="pointer-events-auto mx-auto mt-1 flex max-w-3xl flex-wrap items-center gap-x-3 gap-y-1 px-2 text-xs text-muted-foreground"
		data-testid="memt-memory-bar"
	>
		<label class="flex items-center gap-1.5">
			<span class="font-medium">Memory (person):</span>
			<input
				bind:value={draft}
				class="h-6 w-36 rounded border border-border bg-background px-1.5 text-foreground"
				list="memt-memory-ids"
				onblur={apply}
				onkeydown={(e) => {
					if (e.key === 'Enter') {
						e.preventDefault();
						apply();
					}
				}}
				placeholder="off"
				title="Memory id. Empty = no memory. A new name starts an empty memory."
			/>
			<datalist id="memt-memory-ids">
				{#each memoryStore.ids as id (id)}
					<option value={id}></option>
				{/each}
			</datalist>
		</label>

		{#if memoryStore.id}
			<label
				class="flex items-center gap-1"
				title="Write each turn into this person's memory. Then only the new message is sent: the memory holds the rest."
			>
				<input
					checked={memoryStore.save}
					onchange={(e) => memoryStore.setSave(e.currentTarget.checked)}
					type="checkbox"
				/>
				<span>Save to memory</span>
			</label>

			<button
				class="underline"
				onclick={() => {
					draft = '';
					apply();
				}}
				type="button">off</button
			>

			<span>
				{memoryStore.save ? 'reads and saves' : 'reads only'}: <b>{memoryStore.id}</b>
			</span>
		{/if}
	</div>
{/if}
