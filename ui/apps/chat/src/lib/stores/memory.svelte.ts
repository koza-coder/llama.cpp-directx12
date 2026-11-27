/**
 * memoryStore - MemT native memory (models with a memory branch, see the server README "Native memory")
 *
 * The person (memory id) and the "save to memory" switch, kept in localStorage.
 * `available` is set from GET /memory: the server answers 200 only for a model with native memory.
 */

import { browser } from '$app/env';
import { getAuthHeaders } from '#lib/utils/api-headers.ts';
import { withBase } from '#lib/utils/base-path.ts';

const ID_KEY = 'llama-memt-memory-id';
const SAVE_KEY = 'llama-memt-memory-save';

class MemoryStore {
	available = $state(false);
	ids = $state<string[]>([]);
	id = $state(browser ? (localStorage.getItem(ID_KEY) ?? '') : '');
	save = $state(browser ? localStorage.getItem(SAVE_KEY) === '1' : false);

	setId(value: string) {
		this.id = value.trim();
		localStorage.setItem(ID_KEY, this.id);
	}

	setSave(value: boolean) {
		this.save = value;
		localStorage.setItem(SAVE_KEY, value ? '1' : '0');
	}

	async refresh() {
		try {
			const res = await fetch(withBase('/memory'), { headers: getAuthHeaders() });

			if (!res.ok) {
				this.available = false;

				return;
			}

			const data = await res.json();

			this.ids = (data.memories ?? []).map((m: { id: string }) => m.id);
			this.available = true;
		} catch {
			this.available = false;
		}
	}

	/** Request fields for a chat turn (plain and tool-calling sends); commit when "Save to memory" is on. */
	options(): { memoryId?: string; memoryCommit?: boolean } {
		if (!this.available || !this.id) return {};

		return { memoryCommit: this.save, memoryId: this.id };
	}
}

export const memoryStore = new MemoryStore();
